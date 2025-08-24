# -*- coding: utf-8 -*-
# facepipe/main.py
"""
Entry-point that wires the Stage-1 pipeline:
Reader -> Detector(s) -> Embedder(s) -> RedisWriter

Logic replicated from the monolithic o5p_3stage_debug.py,
but split into modules and with identical behavior.
"""

import os
import sys
import time
import json
import argparse
import logging
import traceback
from multiprocessing import Process, JoinableQueue, cpu_count, Value

from config import DEFAULTS
from utils.redis_utils import (
    configure_logging,
    connect_redis,
    redis_cleanup,
    redis_writer_process,
)
from pipeline.reader import reader_process
from pipeline.detector import detector_process
from pipeline.embedding import embedding_worker




def build_argparser():
    ap = argparse.ArgumentParser(
        description="Reader -> Detectors(SCRFD) -> Embedding(ArcFace) -> RedisWriter (multiprocess) + Redis"
    )

    # IO
    ap.add_argument("--video", type=str, default=DEFAULTS["VIDEO"])
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--save-dir", type=str, default=DEFAULTS["SAVE_DIR"])

    # Modèles
    ap.add_argument("--det-model", type=str, default=DEFAULTS["DET_MODEL"])
    ap.add_argument("--rec-model", type=str, default=DEFAULTS["REC_MODEL"])

    # Détection
    ap.add_argument("--det-input", type=str, default=DEFAULTS["DET_INPUT"])
    ap.add_argument("--det-thresh", type=float, default=DEFAULTS["DET_THRESH"])
    ap.add_argument("--nms-thresh", type=float, default=DEFAULTS["NMS_THRESH"])
    ap.add_argument("--min-face", type=int, default=DEFAULTS["MIN_FACE"])

    # Cadencement
    ap.add_argument("--frame-step", type=int, default=DEFAULTS["FRAME_STEP"])

    # Parallélisme
    ap.add_argument("--num-det", type=int, default=DEFAULTS["NUM_DET"])
    ap.add_argument("--num-emb", type=int, default=DEFAULTS["NUM_EMB"])
    ap.add_argument("--q-frames", type=int, default=DEFAULTS["Q_FRAMES"])
    ap.add_argument("--q-faces", type=int, default=DEFAULTS["Q_FACES"])
    ap.add_argument("--q-redis", type=int, default=DEFAULTS["Q_REDIS"])

    # Redis
    ap.add_argument("--redis-host", type=str, default=DEFAULTS["REDIS_HOST"])
    ap.add_argument("--redis-port", type=int, default=DEFAULTS["REDIS_PORT"])
    ap.add_argument("--redis-prefix", type=str, default=DEFAULTS["REDIS_PREFIX"])
    ap.add_argument("--sim-thresh", type=float, default=DEFAULTS["SIM_THRESH"])
    ap.add_argument("--refine-limit", type=int, default=DEFAULTS["REFINE_LIMIT"])

    # Flou (Laplacien)
    ap.add_argument("--blur-enable", action="store_true", default=DEFAULTS["BLUR_ENABLE"])
    ap.add_argument("--blur-var-thresh", type=float, default=DEFAULTS["BLUR_VAR_THRESH"])
    ap.add_argument("--blur-grid", type=int, default=DEFAULTS["BLUR_GRID"])
    ap.add_argument("--blur-grid-min-keep", type=float, default=DEFAULTS["BLUR_GRID_MIN_KEEP"])
    ap.add_argument("--blur-resize", type=int, default=DEFAULTS["BLUR_RESIZE"])

    # Logs
    ap.add_argument("--debug", action="store_true", default=DEFAULTS["DEBUG"])

    return ap



def parse_size(s: str):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 640, 640


def main():
    args = build_argparser().parse_args()
    configure_logging(args.debug, "root")

    det_w, det_h = parse_size(args.det_input)
    det_input = (det_w, det_h)

    save_dir = None
    if args.save_images:
        save_dir = os.path.abspath(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Les crops seront sauvegardés dans: {save_dir}")

    # Connect Redis + cleanup
    r = connect_redis(args.redis_host, args.redis_port)
    if r is not None:
        redis_cleanup(r, args.redis_prefix)

    # Shared counters & timer
    detected_total = Value('i', 0)   # faces detected & queued to embedders
    processed_total = Value('i', 0)  # faces embedded & queued to writer
    saved_total = Value('i', 0)      # crops saved (if --save-images)
    start_ts = Value('d', 0.0)       # first frame enqueue time

    # Queues
    q_frames: JoinableQueue = JoinableQueue(maxsize=args.q_frames)
    q_faces: JoinableQueue  = JoinableQueue(maxsize=args.q_faces)
    q_redis:  JoinableQueue = JoinableQueue(maxsize=args.q_redis)

    # Single Redis writer process (atomic match-or-add)
    p_writer = Process(
        target=redis_writer_process,
        args=(
            q_redis,
            args.redis_host, args.redis_port, args.redis_prefix,
            save_dir, args.sim_thresh, args.refine_limit,
            saved_total, args.debug
        ),
        name="RedisWriter"
    )
    p_writer.start()

    # Embedder workers
    emb_workers = []
    for i in range(args.num_emb):
        p = Process(
            target=embedding_worker,
            args=(i + 1, q_faces, q_redis, args.rec-model if hasattr(args, 'rec-model') else args.rec_model,  # safety
                  processed_total, args.debug),
            name=f"Worker-{i+1}"
        )
        p.start()
        emb_workers.append(p)

    # Detector processes
    det_procs = []
    for i in range(args.num_det):
        p = Process(
            target=detector_process,
            args=(
                i + 1, q_frames, q_faces,
                args.det_model, det_input, args.det_thresh, args.nms_thresh, args.min_face,
                args.blur_enable, args.blur_var_thresh, args.blur_grid, args.blur_grid_min_keep, args.blur_resize,
                detected_total, args.debug
            ),
            name=f"Detector-{i+1}"
        )
        p.start()
        det_procs.append(p)

    # Reader
    p_reader = Process(
        target=reader_process,
        args=(args.video, args.frame_step, q_frames, args.num_det, start_ts, args.debug),
        name="Reader"
    )
    p_reader.start()

    # Synchronize
    q_frames.join()
    for p in det_procs:
        p.join()

    for _ in range(len(emb_workers)):
        q_faces.put(None)
    q_faces.join()
    for p in emb_workers:
        p.join()

    q_redis.put(None)   # stop writer
    q_redis.join()
    p_writer.join()

    p_reader.join()

    # Summary
    elapsed = 0.0
    if start_ts.value > 0.0:
        elapsed = time.time() - start_ts.value

    print("\n===== Résumé d'exécution =====")
    print(f"⏱️  Temps total écoulé (depuis début lecture vidéo) : {elapsed:.2f} s")
    print(f"🧭  Visages détectés (envoyés aux workers) : {detected_total.value}")
    print(f"🧮  Visages traités (embeddings générés)   : {processed_total.value}")
    if save_dir:
        print(f"💾  Visages sauvegardés (fichiers écrits) : {saved_total.value}")

    if r is None:
        print("⚠️  Redis non connecté -> résumé identités limité.")
        return

    try:
        id_count = 0
        face_count = 0
        print("\n📏 Fréquence par identité (Redis) :")
        for key in r.scan_iter(f"{args.redis_prefix}:*"):
            data = json.loads(r.get(key))
            ident = key.split(":")[1]
            cnt = data.get("count", 0)
            face_count += cnt
            id_count += 1
            print(f"  - {ident} : {cnt} apparitions")

        print("\n📊 Statistiques globales (Redis) :")
        print(f"  👥 Nombre total d'identités : {id_count}")
        print(f"  🖼️ Nombre total de visages traités (via Redis) : {face_count}")
    except Exception as e:
        print(f"Erreur stats Redis: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        sys.exit(1)
