# -*- coding: utf-8 -*-
# facepipe/pipeline/embedding.py

import logging
import numpy as np
from multiprocessing import JoinableQueue, Value
from utils.redis_utils import configure_logging, l2norm
from vision.arcface_wrap import load_arcface


def embedding_worker(worker_id: int,
                     q_faces: JoinableQueue,
                     q_redis: JoinableQueue,
                     rec_model_path: str,
                     processed_total: Value,
                     debug: bool):
    """Consumes aligned crops, computes 512-D normalized embeddings, pushes to Redis-writer queue."""
    configure_logging(debug, f"Worker-{worker_id}")
    log = logging.getLogger(f"Worker-{worker_id}")

    log.info(f"Chargement ArcFace: {rec_model_path}")
    try:
        rec = load_arcface(rec_model_path)
    except Exception:
        log.error("Echec chargement ArcFace", exc_info=True)
        # Drain to avoid blocking join()
        while True:
            try:
                item = q_faces.get_nowait()
            except Exception:
                break
            q_faces.task_done()
        return

    total = 0
    while True:
        item = q_faces.get()
        if item is None:
            q_faces.task_done()
            break

        try:
            fid, aligned = item
            if aligned is None or aligned.size == 0:
                q_faces.task_done()
                continue

            feat = rec.get_feat(aligned)  # 112x112 BGR expected
            if feat is None:
                q_faces.task_done()
                continue

            emb = l2norm(feat.astype(np.float32).reshape(-1))
            # hand-over to single Redis writer
            q_redis.put({
                "frame_id": int(fid),
                "embedding": emb.tolist(),   # JSON-serializable
                "aligned": aligned           # 112x112x3 small image
            })
            total += 1
            with processed_total.get_lock():
                processed_total.value += 1
            q_faces.task_done()
        except Exception:
            try:
                q_faces.task_done()
            except Exception:
                pass

    log.info(f"Worker-{worker_id} terminé. faces_in={total}")
