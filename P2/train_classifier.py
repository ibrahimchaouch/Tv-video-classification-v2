# -*- coding: utf-8 -*-
"""
P2 - Entraînement d'un classifieur (embeddings ArcFace -> nom)
- Lit face-dataset/ (chaque dossier = label)
- Détecte + aligne (SCRFD + norm_crop)
- Extrait embeddings (ArcFace .onnx, 512D L2)
- Entraîne un classifieur linéaire (LogisticRegression)
- Sauvegarde: modèle + label_map + méta
"""

import os
import glob
import json
import argparse
import logging
import numpy as np
import cv2

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Réutilise tes wrappers/Helpers P1
from vision.scrfd_wrap import build_scrfd, scrfd_detect_on_frame  # SCRFD:contentReference[oaicite:9]{index=9}
from insightface.utils.face_align import norm_crop                 # Alignement:contentReference[oaicite:10]{index=10}
from vision.arcface_wrap import load_arcface                      # ArcFace ONNX:contentReference[oaicite:11]{index=11}
from utils.redis_utils import l2norm                              # L2-normalisation:contentReference[oaicite:12]{index=12}


def parse_size(s: str):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 640, 640


def align_face(det, img_bgr, det_thresh, det_input, min_face=40):
    """Détecte + aligne 1 visage. Retourne crop 112x112 (ou None)."""
    bboxes, kpss = scrfd_detect_on_frame(det, img_bgr, det_thresh, det_input, max_num=0)
    if bboxes is None or bboxes.shape[0] == 0:
        return None
    # plus grand visage
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    idx = int(np.argmax(areas))
    kps = None if kpss is None or len(kpss) <= idx else kpss[idx]
    if kps is not None and kps.shape == (5, 2):
        return norm_crop(img_bgr, landmark=kps)
    # fallback: simple crop redimensionné si pas de kps
    x1, y1, x2, y2, _ = bboxes[idx].astype(np.int32)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    if (x2 - x1) < min_face or (y2 - y1) < min_face:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (112, 112))


def build_dataset(root_dir, det_model, rec_model, det_input="640x640",
                  det_thresh=0.7, nms_thresh=0.4, min_face=40, limit_per_class=None):
    """Parcourt face-dataset et produit (X embeddings, y noms)."""
    det_w, det_h = parse_size(det_input)
    det = build_scrfd(det_model, input_size=(det_w, det_h), nms_thresh=nms_thresh)
    rec = load_arcface(rec_model)  # ctx_id=-1 dans wrapper:contentReference[oaicite:13]{index=13}

    X, y = [], []
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    for label in classes:
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            img_paths.extend(glob.glob(os.path.join(root_dir, label, ext)))
        img_paths = sorted(img_paths)[:limit_per_class] if limit_per_class else sorted(img_paths)
        kept = 0
        for p in img_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            aligned = align_face(det, img, det_thresh, (det_w, det_h), min_face=min_face)
            if aligned is None:
                continue
            feat = rec.get_feat(aligned)  # 112x112 BGR → 512D
            if feat is None:
                continue
            emb = l2norm(feat.astype(np.float32).reshape(-1))  # L2:contentReference[oaicite:14]{index=14}
            X.append(emb)
            y.append(label)
            kept += 1
        logging.info(f"{label}: {kept} images retenues")
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y


def train_classifier(X, y, C=12.0, max_iter=2000, test_size=0.2, random_state=42):
    le = LabelEncoder()
    y_idx = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_idx, test_size=test_size, stratify=y_idx, random_state=random_state)

    # LogReg multiclasses sur embeddings normalisés
    clf = LogisticRegression(
        penalty="l2", C=C, max_iter=max_iter, multi_class="auto", n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    if X_te.shape[0] > 0:
        y_pred = clf.predict(X_te)
        print(classification_report(y_te, y_pred, target_names=le.classes_))
    return clf, le


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Chemin vers face-dataset/")
    ap.add_argument("--det-model", required=True, help="SCRFD .onnx")
    ap.add_argument("--rec-model", required=True, help="ArcFace .onnx")
    ap.add_argument("--det-input", default="640x640")
    ap.add_argument("--det-thresh", type=float, default=0.7)
    ap.add_argument("--nms-thresh", type=float, default=0.4)
    ap.add_argument("--min-face", type=int, default=40)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--C", type=float, default=12.0)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out-clf", default="p2_clf.joblib")
    ap.add_argument("--out-labels", default="p2_labels.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    X, y = build_dataset(
        root_dir=args.dataset,
        det_model=args.det_model,
        rec_model=args.rec_model,
        det_input=args.det_input,
        det_thresh=args.det_thresh,
        nms_thresh=args.nms_thresh,
        min_face=args.min_face,
        limit_per_class=(args.limit_per_class or None)
    )
    print(f"Dataset: X={X.shape}, y={len(y)} classes={len(np.unique(y))}")
    clf, le = train_classifier(
        X, y, C=args.C, max_iter=args.max_iter, test_size=args.test_size
    )

    joblib.dump(clf, args.out_clf)
    with open(args.out_labels, "w", encoding="utf-8") as f:
        json.dump({"classes": le.classes_.tolist()}, f, ensure_ascii=False, indent=2)
    print(f"✅ Sauvé: {args.out_clf}, {args.out_labels}")


if __name__ == "__main__":
    main()
