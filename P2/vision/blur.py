# -*- coding: utf-8 -*-
# facepipe/vision/blur.py

import cv2
import numpy as np


def _lap_var(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def _prep_gray(frame_bgr: np.ndarray, resize_w: int) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if resize_w and resize_w > 0 and gray.shape[1] > resize_w:
        r = resize_w / float(gray.shape[1])
        nh = max(1, int(round(gray.shape[0] * r)))
        gray = cv2.resize(gray, (resize_w, nh), interpolation=cv2.INTER_AREA)
    return gray


def is_frame_blurry(frame_bgr: np.ndarray,
                    var_thresh: float,
                    grid: int = 0,
                    min_keep: float = 0.4,
                    resize_w: int = 640):
    """
    Compute Laplacian variance globally (and optionally on an NxN grid).
    Returns (is_blur: bool, metrics: dict[var, ratio, grid]).
    """
    gray = _prep_gray(frame_bgr, resize_w)
    v = _lap_var(gray)

    if v < float(var_thresh):
        return True, {'var': v, 'ratio': 0.0, 'grid': 0}

    if grid and grid > 0:
        h, w = gray.shape[:2]
        gh = max(1, h // grid)
        gw = max(1, w // grid)
        tiles = 0
        sharp = 0
        for r in range(0, h, gh):
            for c in range(0, w, gw):
                r2 = min(h, r + gh)
                c2 = min(w, c + gw)
                tile = gray[r:r2, c:c2]
                if tile.size == 0:
                    continue
                tiles += 1
                if _lap_var(tile) >= float(var_thresh):
                    sharp += 1
        ratio = (sharp / float(tiles)) if tiles > 0 else 0.0
        if ratio < float(min_keep):
            return True, {'var': v, 'ratio': ratio, 'grid': grid}
        return False, {'var': v, 'ratio': ratio, 'grid': grid}

    return False, {'var': v, 'ratio': 1.0, 'grid': 0}
