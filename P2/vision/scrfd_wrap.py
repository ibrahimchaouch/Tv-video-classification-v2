# -*- coding: utf-8 -*-
# facepipe/vision/scrfd_wrap.py

from insightface.model_zoo import get_model


def build_scrfd(det_model_path: str, input_size, nms_thresh: float):
    det = get_model(det_model_path)  # SCRFD
    det.prepare(ctx_id=-1, input_size=input_size, nms_thresh=nms_thresh)  # CPU
    return det


def scrfd_detect_on_frame(det, frame_bgr, det_thresh: float, input_size, max_num: int = 0):
    """
    Compatible with insightface 0.7.x detect() signature variants.
    Returns (bboxes, kpss)
    """
    try:
        return det.detect(frame_bgr, float(det_thresh), input_size, int(max_num))
    except TypeError:
        pass
    try:
        return det.detect(frame_bgr, input_size, float(det_thresh), int(max_num))
    except TypeError:
        pass
    bboxes, kpss = det.detect(frame_bgr, input_size=input_size)
    if bboxes is not None and bboxes.shape[0] > 0:
        keep = bboxes[:, 4] >= float(det_thresh)
        bboxes = bboxes[keep]
        if kpss is not None:
            kpss = kpss[keep]
        if int(max_num) > 0 and bboxes.shape[0] > int(max_num):
            bboxes = bboxes[:int(max_num)]
            if kpss is not None:
                kpss = kpss[:int(max_num)]
    return bboxes, kpss
