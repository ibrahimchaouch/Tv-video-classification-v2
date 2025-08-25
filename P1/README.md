# facepipe — Stage‑1 Face Frequency & Embedding Pipeline

*A modular, reproducible implementation of the first stage (Face Appearance Frequency Analysis)*

> **Scope**: This repository implements **Stage‑1** only — extracting **per‑identity appearance frequency** and a **512‑D face embedding** bank from a TV video. Stages 2–3 (name assignment and web biography retrieval + final genre classification) are out of scope here and discussed in P2 & P3” and the complete pipeline motivation are in the paper; this repo focuses on the engineering details that make Stage‑1 precise and fast.

---

## ✨ Highlights

* **Reader → Detector(SCRFD) → Aligner → Embedder(ArcFace)** with **multiprocessing** and back‑pressure queues.
* **Laplacian‑based blur gate** applied **before detection** (global or grid mode) to skip low‑quality frames.
* **Identity clustering with online centroid update** and **cosine similarity** in normalized 512‑D space.
* **Redis‑backed identity store** with **atomic match‑or‑add** via a dedicated single writer process.
* Deterministic counters & run summary (elapsed time, detected/processed/saved faces, per‑identity counts).

---

## 🧠 Method Summary

### 1) Pre‑filter (Frame Quality)

We compute the variance of the Laplacian on the gray frame (optionally in an $N\times N$ grid).

* Global metric: $v = \mathrm{Var}(\nabla^2 I)$. If $v < T$ the frame is skipped.
* Grid metric: a frame is kept if at least a fraction $\rho$ of tiles satisfy $\mathrm{Var}(\nabla^2 I_{tile}) \ge T$.

This gate eliminates blurry frames before detection and reduces false positives and wasted compute.

### 2) Detection (SCRFD, ONNX)

We use InsightFace’s **SCRFD** detector (`det_500m.onnx`) prepared on CPU. Input size is configurable (e.g., `640x640`). We forward the original BGR frame; for each detection whose width ≥ `--min-face`, we keep its 5‑point landmarks (when available).

### 3) Alignment

If 5‑point landmarks are available, we apply InsightFace’s `norm_crop` to produce a **112×112** aligned face; otherwise we fall back to a tight box crop resized to 112×112.

### 4) Embedding (ArcFace, ONNX)

We compute a 512‑D feature with **ArcFace** (`w600k_mbf.onnx`) and **L2‑normalize** it:

$$
\tilde{\mathbf{f}} = \frac{\mathbf{f}}{\lVert\mathbf{f}\rVert_2 + \varepsilon}\,,\qquad \tilde{\mathbf{f}}\in\mathbb{R}^{512},\; \lVert\tilde{\mathbf{f}}\rVert_2=1.
$$

### 5) Identity Matching with Online Centroid

Each Redis record stores at least `{embedding, centroid, refine_count, count}`. Given a new normalized feature $\tilde{\mathbf{f}}$, we search for the best key by cosine similarity $s = \tilde{\mathbf{c}}^\top\tilde{\mathbf{f}}$. If $s \ge \tau$ we **merge** into that identity; else we **create** a new one.

Centroid update (bounded by `--refine-limit`) uses a running mean **then re‑normalizes**:


$$
\mathbf{c}_{t+1}
= \mathrm{norm}\!\left(\frac{r_t\,\mathbf{c}_t + \tilde{\mathbf{f}}}{r_t + 1}\right),
\qquad
r_{t+1} = \min\!\bigl(r_t + 1,\, \text{\texttt{refine\_limit}}\bigr).
$$

Frequency is tracked via `count ← count + 1` at each merge. This field is central to downstream stages.

### 6) Atomicity & Concurrency

All **match‑or‑add** operations are **serialized** through a **dedicated Redis writer process** fed by a queue. This design ensures atomicity without per‑key locks in readers. The writer performs the best‑match search and the update/create in one critical section per request.

---

## 🗂️ Repository Layout

```
facepipe/
├─ README.md
├─ requirements.txt
├─ config.py                  # defaults (paths, thresholds, ports)
├─ main.py                    # orchestrates processes & queues
├─ video/
│   └─ v1.mp4                 # sample (not in git)
├─ models/                    # optional local notes
├─ scripts/
│   └─ start_redis.sh         # start redis-server; flushall
├─ pipeline/
│   ├─ __init__.py
│   ├─ reader.py              # frame reader process
│   ├─ detector.py            # detector process (SCRFD + blur gate + align)
│   ├─ embedding.py           # embedder workers
│   └─ redis_writer.py        # single writer (atomic match_or_add)
├─ vision/
│   ├─ __init__.py
│   ├─ blur.py                # Laplacian metrics (global & grid)
│   ├─ scrfd_wrap.py          # helper around insightface SCRFD
│   └─ arcface_wrap.py        # helper around ArcFace ONNX
└─ utils/
    ├─ __init__.py
    └─ redis_utils.py         # connect/cleanup helpers
```

---

## 🔧 Installation

1. **Python env**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. **Models**
   Install `insightface==0.7.3`. At first run, it will download the models automatically into `~/.insightface/models/models/buffalo_s/`:

* `det_500m.onnx` (SCRFD detector)
* `w600k_mbf.onnx` (ArcFace embedder)

3. **Redis**
   Make sure `redis-server` is available in `$PATH`.

---

## ▶️ Running

Start Redis (the script will also `FLUSHALL`):

```bash
chmod +x scripts/start_redis.sh
./scripts/start_redis.sh 6381
```

Run the pipeline:

```bash
python main.py \
  --video ./video/v1.mp4 \
  --det-model ~/.insightface/models/models/buffalo_s/det_500m.onnx \
  --rec-model ~/.insightface/models/models/buffalo_s/w600k_mbf.onnx \
  --det-input 640x640 --det-thresh 0.7 --min-face 60 \
  --frame-step 15 --num-det 2 --num-emb 6 \
  --redis-host localhost --redis-port 6381 --sim-thresh 0.27 --refine-limit 30 \
  --blur-enable --blur-var-thresh 120 --blur-grid 0 \
  --save-images --save-dir ./captures_faces \
  --debug
```

> If `--debug` is omitted, only the progress bars and the final summary are shown.

---

## 🧰 Command‑line Arguments (selected)

* **IO**: `--video`, `--save-images`, `--save-dir`
* **Detector**: `--det-model`, `--det-input=LxH`, `--det-thresh`, `--nms-thresh`, `--min-face`
* **Scheduling**: `--frame-step`
* **Parallelism**: `--num-det`, `--num-emb`, `--q-frames`, `--q-faces`
* **Redis**: `--redis-host`, `--redis-port`, `--redis-prefix`, `--sim-thresh`, `--refine-limit`
* **Blur gate**: `--blur-enable`, `--blur-var-thresh`, `--blur-grid`, `--blur-grid-min-keep`, `--blur-resize`
* **Logging**: `--debug` (verbose) / default (quiet)

---

## 📊 Output & Summary

At the end of a run the program prints:

* **Elapsed time** (from the first enqueued frame),
* **Faces detected → queued**, **faces embedded**, **faces saved**,
* **Per‑identity frequency** and a **global tally** read back from Redis (ground truth for Stage‑1).

Saved crops (if `--save-images`) are organized as:

```
./captures_faces/
  ├─ <identity_id_A>/ frame_01234_XXXX.jpg …
  ├─ <identity_id_B>/ …
  └─ …
```

---

## 🧪 Accuracy‑preserving performance tips

* Keep `--det-input` at **640×640** unless profiling shows a safe reduction.
* Increase `--frame-step` to skip frames uniformly when working with long videos.
* Use **multiple detector processes** (I/O bound) and **more embedder workers** (CPU bound).
* Use the **blur gate** to avoid spending compute on uninformative frames.
* Tune `--sim-thresh` around **0.25–0.30** with your data; too low merges distinct identities, too high splits.

---

## 🔒 Notes on Atomicity

* All **match‑or‑add** operations go through a **single writer**. This ensures that the best‑match search and the subsequent update/create happen as one serialized transaction per face.
* The writer is also responsible for updating the **frequency counter** (`count`) and the **centroid** with re‑normalization.

---

## ✅ Reproducibility Notes

* We pin **`insightface==0.7.3`** to preserve the detector/ONNX call signatures.
* Models are CPU by default (`ctx_id=-1`). If a GPU provider is configured in ONNX Runtime, you can adapt the wrappers.

---

## 📦 Requirements

See `requirements.txt`. Minimal set:

```
insightface==0.7.3
onnxruntime>=1.16
opencv-python>=4.8
numpy>=1.24
scipy>=1.10
scikit-learn
redis>=5.0
tqdm
```

---

## 📜 License

MIT. See `LICENSE`.

---

## ✍️ Citation

If you find this useful, please cite the paper (details in the repository or the project page).
