# -*- coding: utf-8 -*-
# facepipe/pipeline/reader.py

import time
import logging
import cv2
from tqdm import tqdm
from multiprocessing import JoinableQueue, Value
from utils.redis_utils import configure_logging


def reader_process(video_path: str,
                   frame_step: int,
                   q_frames: JoinableQueue,
                   num_det: int,
                   start_ts: Value,
                   debug: bool):
    """Reads frames, down-samples by frame_step, sends to detectors."""
    configure_logging(debug, "Reader")
    log = logging.getLogger("Reader")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Impossible d'ouvrir la vidéo: {video_path}")
        for _ in range(num_det):
            q_frames.put(None)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=total, desc="🎞️ Lecture vidéo", unit="frame")

    fid = -1
    sent = 0
    first_sent = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fid += 1
            pbar.update(1)
            if frame_step > 1 and (fid % frame_step) != 0:
                continue
            if not first_sent:
                with start_ts.get_lock():
                    if start_ts.value == 0.0:
                        start_ts.value = time.time()
                first_sent = True
            q_frames.put((fid, frame))
    finally:
        cap.release()
        pbar.close()
        for _ in range(num_det):
            q_frames.put(None)
            sent += 1
        log.info(f"Reader: fin. Frames envoyées={fid+1}, sentinelles={sent}")
