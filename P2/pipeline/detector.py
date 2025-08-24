# -*- coding: utf-8 -*-
# facepipe/pipeline/detector.py

import logging
import numpy as np
import cv2
from multiprocessing import JoinableQueue, Value
from insightface.utils.face_align import norm_crop

from utils.redis_utils import configure_logging
from vision.blur import is_frame_blurry
from vision.scrfd_wrap import build_scrfd, scrfd_detect_on_frame


def detector_process(det_id: int,
                     q_frames: JoinableQueue,
                     q_faces: JoinableQueue,
                     det_model_path: str,
                     det_input,
                     det_thresh: float,
                     nms_thresh: float,
                     min_face: int,
                     blur_enable: bool,
                     blur_var_thresh: float,
                     blur_grid: int,
                     blur_grid_min_keep: float,
                     blur_resize_w: int,
                     detected_total: Value,
                     debug: bool):
    """Runs SCRFD on frames, applies blur gate BEFORE detection, aligns faces, pushes crops to embedder queue."""
    configure_logging(debug, f"Detector-{det_id}")
    log = logging.getLogger(f"Detector-{det_id}")

    log.info(f"Init SCRFD: {det_model_path} | input={det_input} | det_thresh={det_thresh} | nms={nms_thresh}")
    det = build_scrfd(det_model_path, det_input, nms_thresh)

    total_in = 0
    total_out = 0
    skipped_blur = 0

    while True:
        item = q_frames.get()
        try:
            if item is None:
                q_frames.task_done()
                break
            fid, frame = item
            total_in += 1

            # Blur filter BEFORE detection
            if blur_enable:
                is_blur, m = is_frame_blurry(
                    frame_bgr=frame,
                    var_thresh=blur_var_thresh,
                    grid=blur_grid,
                    min_keep=blur_grid_min_keep,
                    resize_w=blur_resize_w
                )
                if is_blur:
                    skipped_blur += 1
                    log.info(f"Frame {fid} ignorée (floue). var={m['var']:.1f} | ratio_net={m['ratio']:.2f} | grid={m['grid']}")
                    q_frames.task_done()
                    continue

            try:
                bboxes, kpss = scrfd_detect_on_frame(det, frame, det_thresh, det_input, max_num=0)
            except Exception:
                q_frames.task_done()
                continue

            if bboxes is None or bboxes.shape[0] == 0:
                q_frames.task_done()
                continue

            kept = 0
            for i in range(bboxes.shape[0]):
                x1, y1, x2, y2, score = bboxes[i].astype(np.float32)
                w = x2 - x1
                if w < float(min_face):
                    continue
                kept += 1

                kps = kpss[i] if (kpss is not None and len(kpss) > i) else None
                if kps is not None and kps.shape == (5, 2):
                    aligned = norm_crop(frame, landmark=kps)
                else:
                    x1i, y1i = max(0, int(x1)), max(0, int(y1))
                    x2i, y2i = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                    crop = frame[y1i:y2i, x1i:x2i]
                    aligned = cv2.resize(crop, (112, 112)) if crop.size > 0 else None

                if aligned is not None and aligned.size > 0:
                    q_faces.put((fid, aligned))
                    total_out += 1
                    with detected_total.get_lock():
                        detected_total.value += 1

            if kept > 0:
                log.info(f"Frame {fid}: {kept} visage(s) -> faces_queue")
            q_frames.task_done()
        except Exception:
            try:
                q_frames.task_done()
            except Exception:
                pass

    log.info(f"Detector-{det_id} terminé. frames_in={total_in}, faces_out={total_out}, skipped_blur={skipped_blur}")
