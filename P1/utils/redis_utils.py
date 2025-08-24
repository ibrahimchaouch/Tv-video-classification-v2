# -*- coding: utf-8 -*-
# facepipe/utils/redis_utils.py
"""
Redis helpers + math helpers + logging + SINGLE writer process.
Implements atomic match-or-add.
"""

import os
import json
import uuid
import logging
import numpy as np
from typing import Optional, Tuple
from multiprocessing import JoinableQueue, Value

try:
    import redis as _redis
except Exception:
    _redis = None

# -------- Helpers --------
def _counter_key(prefix: str) -> str:
    # Important: underscore (pas de ':') pour ne PAS matcher f"{prefix}:*"
    return f"{prefix}_counter"

# -------- Logging --------
def configure_logging(debug: bool, name: Optional[str] = None):
    level = logging.INFO if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s - %(levelname)s - %(message)s" if debug else "%(message)s"
    )
    if name:
        log = logging.getLogger(name)
        log.setLevel(level)

# -------- Math helpers --------
def l2norm(x: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# -------- Redis connect/cleanup --------
def connect_redis(host: str, port: int):
    if _redis is None:
        return None
    try:
        r = _redis.Redis(host=host, port=port, decode_responses=True)
        r.ping()
        logging.info(f"Redis OK sur {host}:{port}")
        return r
    except Exception as e:
        logging.warning(f"Redis indisponible: {e}")
        return None

def redis_cleanup(r, prefix: str):
    if r is None:
        return
    try:
        cnt = 0
        for k in r.scan_iter(f"{prefix}:*"):
            r.delete(k)
            cnt += 1
        # supprimer aussi le compteur pour repartir à identity_1 au prochain run
        try:
            r.delete(_counter_key(prefix))
        except Exception:
            pass
        logging.info(f"Redis nettoyé ({prefix}:*) -> {cnt} clés supprimées")
    except Exception as e:
        logging.warning(f"Impossible de nettoyer Redis: {e}")

# -------- Identity match-or-add --------
def match_or_add_face(r,
                      prefix: str,
                      emb: np.ndarray,
                      sim_thresh: float,
                      refine_limit: int,
                      save_dir: Optional[str],
                      frame_id: int,
                      aligned_face: Optional[np.ndarray]) -> Tuple[str, bool]:
    """
    Search best identity by cosine in normalized space, update centroid & count if matched,
    otherwise create a new identity. Optionally save the aligned crop.
    Returns (identity_id, saved_ok).
    """
    identity_id = None
    best_key, best_data, best_score = None, None, -1.0
    saved_ok = False

    # Find best match (robuste: ignorer les clés non conformes)
    if r is not None:
        for key in r.scan_iter(f"{prefix}:*"):  # ne voit PAS le compteur (_counter)
            try:
                raw = r.get(key)
                if not raw:
                    continue
                data = json.loads(raw)
                if not isinstance(data, dict):
                    continue
                vec = data.get("centroid") or data.get("embedding")
                if vec is None:
                    continue
                ref = l2norm(np.array(vec, dtype=np.float32))
                score = cos_sim(ref, emb)
                if score > best_score:
                    best_score = score
                    best_key, best_data = key, data
            except Exception:
                # ne pas casser la boucle si une clé est invalide
                continue

    # Update or create
    if best_score >= sim_thresh and best_key is not None:
        identity_id = best_key.split(":")[1]
        if "centroid" in best_data and best_data.get("refine_count", 0) < refine_limit:
            c = np.array(best_data["centroid"], dtype=np.float32)
            rc = best_data.get("refine_count", 1)
            upd = l2norm((c * rc + emb) / (rc + 1))
            best_data["centroid"] = upd.tolist()
            best_data["refine_count"] = rc + 1
        best_data["count"] = best_data.get("count", 0) + 1
        if r is not None:
            try:
                r.set(best_key, json.dumps(best_data))
            except Exception:
                pass
    else:
        # Générer un ID lisible, incrémental, sans polluer le scan f"{prefix}:*"
        if r is not None:
            try:
                new_id = r.incr(_counter_key(prefix))  # atomic
                identity_id = f"identity_{new_id}"
            except Exception:
                identity_id = None
        if identity_id is None:
            # fallback lisible mais non incrémental si Redis indispo
            identity_id = f"identity_{uuid.uuid4().hex[:6]}"

        rec = {
            "embedding": emb.tolist(),
            "centroid": emb.tolist(),
            "refine_count": 1,
            "count": 1
        }
        if r is not None:
            try:
                r.set(f"{prefix}:{identity_id}", json.dumps(rec))
            except Exception:
                pass

    # Optional save (dossier = identity_id)
    if save_dir is not None and aligned_face is not None and aligned_face.size > 0:
        try:
            ident_dir = os.path.join(save_dir, identity_id)
            os.makedirs(ident_dir, exist_ok=True)
            name = f"frame_{frame_id}_{uuid.uuid4().hex[:8]}.jpg"
            import cv2
            saved_ok = cv2.imwrite(os.path.join(ident_dir, name), aligned_face)
        except Exception:
            saved_ok = False

    return identity_id, saved_ok

# -------- Single Redis writer process --------
def redis_writer_process(q_redis: JoinableQueue,
                         redis_host: str,
                         redis_port: int,
                         redis_prefix: str,
                         save_dir: Optional[str],
                         sim_thresh: float,
                         refine_limit: int,
                         saved_total: Value,
                         debug: bool):
    """
    Sole process that serializes all identity match-or-add ops.
    This preserves atomicity (no per-key locks needed elsewhere).
    """
    configure_logging(debug, "RedisWriter")
    log = logging.getLogger("RedisWriter")

    r = connect_redis(redis_host, redis_port)
    if r is None:
        log.warning("Redis non disponible : le writer consommera quand même la queue sans écrire.")
    else:
        log.info(f"Writer connecté à Redis {redis_host}:{redis_port}")

    total_in = 0
    total_saved = 0

    while True:
        item = q_redis.get()
        if item is None:
            q_redis.task_done()
            break
        try:
            fid = int(item["frame_id"])
            emb = np.array(item["embedding"], dtype=np.float32)
            aligned = item["aligned"]

            _, saved_ok = match_or_add_face(r, redis_prefix, emb, sim_thresh, refine_limit,
                                            save_dir, fid, aligned)
            total_in += 1
            if saved_ok:
                total_saved += 1
                with saved_total.get_lock():
                    saved_total.value += 1
            q_redis.task_done()
        except Exception:
            try:
                q_redis.task_done()
            except Exception:
                pass

    log.info(f"RedisWriter terminé. items_in={total_in}, saved={total_saved}")
