# -*- coding: utf-8 -*-
# facepipe/config.py
import os
from multiprocessing import cpu_count

BASE = os.path.dirname(__file__)

# Valeurs par défaut centralisées pour TOUS les arguments CLI
DEFAULTS = {
    # IO
    "VIDEO": os.path.join(BASE, "video", "v1.mp4"),
    "SAVE_DIR": os.path.join(BASE, "captures_faces"),

    # Modèles
    "DET_MODEL": os.path.expanduser("~/.insightface/models/models/buffalo_s/det_500m.onnx"),
    "REC_MODEL": os.path.expanduser("~/.insightface/models/models/buffalo_s/w600k_mbf.onnx"),

    # Détection
    "DET_INPUT": "640x640",   # string "LxH"
    "DET_THRESH": 0.7,
    "NMS_THRESH": 0.4,
    "MIN_FACE": 60,

    # Cadencement
    "FRAME_STEP": 15,         # 1 frame sur N

    # Parallélisme
    "NUM_DET": max(1, cpu_count() // 2),
    "NUM_EMB": max(1, cpu_count() - 1),
    "Q_FRAMES": 64,           # taille max queue frames
    "Q_FACES": 256,           # taille max queue faces
    "Q_REDIS": 1024,          # taille max queue writer Redis

    # Redis
    "REDIS_HOST": "localhost",
    "REDIS_PORT": 6381,
    "REDIS_PREFIX": "face",
    "SIM_THRESH": 0.23,
    "REFINE_LIMIT": 30,

    # Filtre flou (Laplacien)
    "BLUR_ENABLE": False,
    "BLUR_VAR_THRESH": 120.0,
    "BLUR_GRID": 0,
    "BLUR_GRID_MIN_KEEP": 0.4,
    "BLUR_RESIZE": 640,

    # Logs
    "DEBUG": False,
}
