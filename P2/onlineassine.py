# -*- coding: utf-8 -*-
"""
P2 NameAssigner Daemon
- Ecoute Redis (keyevent 'set') pour clés prefix:identity_*
- Fallback: polling périodique si notifications indisponibles
- Déclenche la prédiction en fonction de refine_count / count / seuils
- Ecrit {name, confidence, pred_ts} dans la même clé (optionnel)

Dépendances : redis-py, numpy, joblib (ou onnxruntime si .onnx)
"""

import os
import time
import json
import argparse
import logging
import numpy as np

from utils.redis_utils import connect_redis, l2norm  # decode_responses=True, L2 helper:contentReference[oaicite:9]{index=9}

# ---------- Modèle: sklearn .joblib OU ONNX ----------
class Predictor:
    def __init__(self, clf_path: str, labels_path: str):
        self.labels = json.load(open(labels_path, "r", encoding="utf-8"))["classes"]
        self.is_onnx = clf_path.lower().endswith(".onnx")
        if self.is_onnx:
            try:
                import onnxruntime as ort
            except Exception as e:
                raise RuntimeError("onnxruntime requis pour .onnx") from e
            self.sess = ort.InferenceSession(clf_path, providers=["CPUExecutionProvider"])
            outs = [o.name for o in self.sess.get_outputs()]
            # compat: skl2onnx => "probabilities", export softmax pur => "probas"
            self.out_name = "probas" if "probas" in outs else ("probabilities" if "probabilities" in outs else outs[-1])
            self.in_name = self.sess.get_inputs()[0].name
            self.kind = "onnx"
        else:
            import joblib
            self.clf = joblib.load(clf_path)
            self.kind = "sklearn"

    def predict(self, x512: np.ndarray):
        x512 = x512.astype(np.float32).reshape(1, -1)  # 1x512
        if self.kind == "onnx":
            probas = self.sess.run([self.out_name], {self.in_name: x512})[0]  # (1,K)
            probs = probas[0]
        else:
            if hasattr(self.clf, "predict_proba"):
                probs = self.clf.predict_proba(x512)[0]
            else:
                k = int(self.clf.predict(x512)[0])
                probs = np.zeros(len(self.labels), dtype=np.float32); probs[k] = 1.0
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx]), probs


# ---------- Logique de décision ----------
def should_predict(data: dict,
                   min_refine: int,
                   min_count: int,
                   finalize_on_limit: bool,
                   refine_limit: int) -> bool:
    # Pas de centroïde → rien à faire (P1 écrit centroid et refine_count/count):contentReference[oaicite:10]{index=10}
    if data.get("centroid") is None and data.get("embedding") is None:
        return False

    name = data.get("name")
    rcount = int(data.get("refine_count", 1))
    cnt = int(data.get("count", 1))

    # Finalisation stricte à refine_limit
    if finalize_on_limit and refine_limit > 0 and rcount >= refine_limit:
        return True

    # Si déjà nommé avec bonne confiance, tu peux refuser de re-prédire (au choix)
    if name and name != "unknown":
        # On ne repropose que si un palier a été franchi (optionnel, géré ailleurs)
        return False

    # Déclencheurs souples
    if rcount >= int(min_refine):
        return True
    if cnt >= int(min_count):
        return True
    return False


def extract_identity_id(key: str, prefix: str) -> str:
    # P1 enregistre sous f"{prefix}:{identity_id}":contentReference[oaicite:11]{index=11}
    try:
        return key.split(f"{prefix}:")[1]
    except Exception:
        return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6381)  # par défaut dans ton projet:contentReference[oaicite:12]{index=12}
    ap.add_argument("--redis-prefix", default="face")
    ap.add_argument("--clf", required=True, help="p2_clf.joblib ou p2_clf.onnx")
    ap.add_argument("--labels", default="p2_labels.json")
    ap.add_argument("--min-conf", type=float, default=0.60, help="seuil probas pour nommer, sinon 'unknown'")
    ap.add_argument("--min-refine", type=int, default=5, help="déclencher à partir de ce refine_count")
    ap.add_argument("--min-count", type=int, default=5, help="ou à partir de ce count")
    ap.add_argument("--refine-limit", type=int, default=30, help="REFINE_LIMIT de P1 (voir config):contentReference[oaicite:13]{index=13}")
    ap.add_argument("--finalize-on-limit", action="store_true", help="forcer prédiction quand refine_count atteint le REFINE_LIMIT")
    ap.add_argument("--write-back", action="store_true", help="écrit name/confidence/pred_ts dans la même clé Redis")
    ap.add_argument("--poll-interval", type=float, default=5.0, help="intervalle polling si notifications indisponibles")
    ap.add_argument("--relabel-step", type=int, default=5, help="re-prédire si refine_count a augmenté d'au moins ce pas")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("NameAssigner")

    predictor = Predictor(args.clf, args.labels)
    log.info(f"Classifieur chargé ({predictor.kind})")

    # Etat local pour éviter de re-traiter les mêmes étapes
    # clé: identity_id -> dernier refine_count traité
    last_refine_seen = {}

    while True:
        r = connect_redis(args.redis_host, args.redis_port)  # ping interne:contentReference[oaicite:14]{index=14}
        if r is None:
            log.warning("Redis indisponible, retry dans 3s…")
            time.sleep(3.0)
            continue

        # Essai d'activer les notifications (si autorisé)
        use_events = False
        try:
            r.config_set("notify-keyspace-events", "KEA")  # Keyevent + All (set inclus)
            use_events = True
            log.info("Keyevent notifications activées (KEA)")
        except Exception:
            log.info("Keyevent notifications non disponibles → fallback polling")

        # Traitement initial : scan des identités existantes
        patt = f"{args.redis_prefix}:*"
        for key in r.scan_iter(patt):
            try:
                raw = r.get(key)
                if not raw:
                    continue
                data = json.loads(raw)
                ident = extract_identity_id(key, args.redis_prefix)
                rcount = int(data.get("refine_count", 1))
                last = last_refine_seen.get(ident, 0)
                # Re-prédire si pas encore nommé OU si on a franchi un palier
                if should_predict(data, args.min_refine, args.min_count, args.finalize_on_limit, args.refine_limit) and (rcount >= last + args.relabel_step or not data.get("name")):
                    centroid = data.get("centroid") or data.get("embedding")
                    x = l2norm(np.array(centroid, dtype=np.float32))  # sécurité L2:contentReference[oaicite:15]{index=15}
                    name, conf, _ = predictor.predict(x)
                    if conf < float(args.min_conf):
                        name = "unknown"
                    log.info(f"[init] {key} → {name} (conf={conf:.3f}, refine={rcount})")
                    last_refine_seen[ident] = rcount
                    if args.write_back:
                        data["name"] = name
                        data["confidence"] = conf
                        data["pred_ts"] = time.time()
                        r.set(key, json.dumps(data))
            except Exception:
                continue

        if use_events:
            # Abonnement aux événements SET
            try:
                db = r.connection_pool.connection_kwargs.get("db", 0)
            except Exception:
                db = 0
            ch = f"__keyevent@{db}__:set"
            ps = r.pubsub(ignore_subscribe_messages=True)
            try:
                ps.psubscribe(ch)
                log.info(f"Abonné aux événements: {ch}")
            except Exception:
                log.info("Echec abonnement → fallback polling")
                use_events = False

        # Boucle principale
        try:
            if use_events:
                for msg in ps.listen():
                    try:
                        key = msg.get("data")
                        if not isinstance(key, str):
                            continue
                        if not key.startswith(f"{args.redis_prefix}:"):
                            continue
                        raw = r.get(key)
                        if not raw:
                            continue
                        data = json.loads(raw)
                        ident = extract_identity_id(key, args.redis_prefix)
                        rcount = int(data.get("refine_count", 1))
                        last = last_refine_seen.get(ident, 0)

                        # Déclenchement précis
                        if should_predict(data, args.min_refine, args.min_count, args.finalize_on_limit, args.refine_limit):
                            # Evite re-prédire si refine_count inchangé et déjà nommé
                            if data.get("name") and rcount <= last:
                                continue
                            if (rcount < last + args.relabel_step) and data.get("name"):
                                continue
                            centroid = data.get("centroid") or data.get("embedding")
                            if centroid is None:
                                continue
                            x = l2norm(np.array(centroid, dtype=np.float32))
                            name, conf, _ = predictor.predict(x)
                            if conf < float(args.min_conf):
                                name = "unknown"
                            log.info(f"[event] {key} → {name} (conf={conf:.3f}, refine={rcount})")
                            last_refine_seen[ident] = rcount
                            if args.write_back:
                                data["name"] = name
                                data["confidence"] = conf
                                data["pred_ts"] = time.time()
                                r.set(key, json.dumps(data))
                    except Exception:
                        # continue à écouter
                        continue
            else:
                # Polling périodique
                while True:
                    time.sleep(max(0.5, float(args.poll_interval)))
                    for key in r.scan_iter(patt):
                        try:
                            raw = r.get(key)
                            if not raw:
                                continue
                            data = json.loads(raw)
                            ident = extract_identity_id(key, args.redis_prefix)
                            rcount = int(data.get("refine_count", 1))
                            last = last_refine_seen.get(ident, 0)
                            if not should_predict(data, args.min_refine, args.min_count, args.finalize_on_limit, args.refine_limit):
                                continue
                            if data.get("name") and rcount <= last:
                                continue
                            if (rcount < last + args.relabel_step) and data.get("name"):
                                continue
                            centroid = data.get("centroid") or data.get("embedding")
                            if centroid is None:
                                continue
                            x = l2norm(np.array(centroid, dtype=np.float32))
                            name, conf, _ = predictor.predict(x)
                            if conf < float(args.min_conf):
                                name = "unknown"
                            log.info(f"[poll] {key} → {name} (conf={conf:.3f}, refine={rcount})")
                            last_refine_seen[ident] = rcount
                            if args.write_back:
                                data["name"] = name
                                data["confidence"] = conf
                                data["pred_ts"] = time.time()
                                r.set(key, json.dumps(data))
                        except Exception:
                            continue
        except KeyboardInterrupt:
            log.info("Arrêt demandé")
            return
        except Exception as e:
            log.warning(f"Connexion perdue ({e}), reconnexion dans 3s…")
            time.sleep(3.0)
            continue


if __name__ == "__main__":
    main()
