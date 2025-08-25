# -*- coding: utf-8 -*-
"""
P2 - Production : associer un nom aux identités P1
- Charge clf + labels
- Connecte Redis, lit prefix:identity_*
- Pour chaque enregistrement: récupère 'centroid' (512D), L2, predict_proba
- Ecrit 'name' + 'confidence' dans la même clé (facultatif)
"""

import json
import argparse
import logging
import numpy as np
import joblib

from utils.redis_utils import connect_redis, l2norm  # Redis helpers:contentReference[oaicite:18]{index=18}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6381)
    ap.add_argument("--redis-prefix", default="face")
    ap.add_argument("--clf", default="p2_clf.joblib")
    ap.add_argument("--labels", default="p2_labels.json")
    ap.add_argument("--min-conf", type=float, default=0.55, help="seuil sur probabilité max")
    ap.add_argument("--write-back", action="store_true", help="écrire name/confidence dans Redis")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Charge classifieur + labels
    clf = joblib.load(args.clf)
    with open(args.labels, "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    # Connect Redis et itère sur les identities
    r = connect_redis(args.redis_host, args.redis_port)  # ping intégré:contentReference[oaicite:19]{index=19}
    if r is None:
        raise RuntimeError("Redis indisponible")

    pattern = f"{args.redis_prefix}:*"
    total = 0
    named = 0
    for key in r.scan_iter(pattern):
        raw = r.get(key)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        centroid = data.get("centroid") or data.get("embedding")
        if centroid is None:
            continue

        x = l2norm(np.array(centroid, dtype=np.float32).reshape(-1))  # sécurité:contentReference[oaicite:20]{index=20}
        # proba sur classes connues
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba([x])[0]
        else:
            # fallback : décision dure
            pred = clf.predict([x])[0]
            probs = np.zeros(len(classes), dtype=np.float32)
            probs[pred] = 1.0

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        name = classes[idx] if conf >= float(args.min_conf) else "unknown"

        print(f"{key}  →  {name}  (conf={conf:.3f})")
        total += 1

        if args.write_back:
            try:
                data["name"] = name
                data["confidence"] = conf
                r.set(key, json.dumps(data))
                named += 1
            except Exception:
                pass

    print(f"\nRésumé: identities lues={total}, mises à jour={named if args.write_back else 0}")


if __name__ == "__main__":
    main()
