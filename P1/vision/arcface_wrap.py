# -*- coding: utf-8 -*-
# facepipe/vision/arcface_wrap.py

from insightface.model_zoo import arcface_onnx


def load_arcface(rec_model_path: str) -> arcface_onnx.ArcFaceONNX:
    """
    ArcFace wrapper for insightface==0.7.3 (no providers/session in ctor).
    """
    rec = arcface_onnx.ArcFaceONNX(model_file=rec_model_path)
    rec.prepare(ctx_id=-1)  # CPU
    return rec
