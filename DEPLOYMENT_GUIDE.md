# Deployment Guide

How to run the deepfake-detection demo locally and (optionally) on
Google Colab or in Docker. The default deployment is **local CPU on
Windows** — that's how the project was developed and tested.

## Local — Windows (PowerShell)

### Prerequisites

- Python 3.10 or 3.11 (3.12+ untested with `facenet-pytorch`).
- Git (optional, only if cloning the repo).
- 2 GB free disk space (model + cached features).
- 4 GB free RAM.

### Setup (one-time)

```powershell
# 1. Create a virtual env (skip if you already have F:\ml_project\.venv)
python -m venv F:\ml_project\.venv

# 2. Activate it
F:\ml_project\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r F:\ml_project\deepfake-detection\requirements.txt
```

### Verify the model checkpoints exist

```powershell
Get-ChildItem F:\ml_project\deepfake-detection\models\
# Expected: cnn_baseline_best.pth, hybrid_best.pth, hybrid_head_only.pth
```

If `cnn_baseline_best.pth` is missing, retrain it (see [TRAINING_GUIDE.md](TRAINING_GUIDE.md))
or restore it from your backup. The Flask app will start without a
checkpoint but the `/predict` endpoint will return `503 model_not_loaded`.

### Run the demo

```powershell
F:\ml_project\.venv\Scripts\python.exe F:\ml_project\deepfake-detection\run_app.py
```

You should see:
```
[YYYY-MM-DD HH:MM:SS] INFO app: Model kind: cnn | decision threshold: 0.750
[YYYY-MM-DD HH:MM:SS] INFO app.predictor: Loaded CNNFramePredictor (device=cpu) from ...\cnn_baseline_best.pth
[YYYY-MM-DD HH:MM:SS] INFO app: Model and preprocessor initialised.
 * Running on http://127.0.0.1:5000
```

Open <http://127.0.0.1:5000> in any browser. Upload an `.mp4`. First
prediction takes ~15 s on this laptop (face detection + 32-frame ResNet-50
forward); subsequent predictions are similar — the model is loaded once
at startup, not per request.

### Sanity probes

```powershell
# Liveness
curl http://127.0.0.1:5000/health
# {"model_loaded":true,"ok":true}

# Predict a video via cURL
curl -X POST -F "video=@F:\path\to\sample.mp4" http://127.0.0.1:5000/predict
```

### Stopping the server

`Ctrl+C` in the terminal. Uploaded files in `app/static/uploads/` are
deleted after each prediction; logs persist under `app/logs/`.

### Configuration via environment variables

```powershell
$env:APP_HOST = "0.0.0.0"        # bind on all interfaces (LAN demo)
$env:APP_PORT = "8080"
$env:APP_MODEL_KIND = "hybrid"   # switch from CNN to hybrid (if hybrid_best.pth is good)
$env:APP_SECRET_KEY = "<random>"
F:\ml_project\.venv\Scripts\python.exe F:\ml_project\deepfake-detection\run_app.py
```

## Local — Linux / macOS

Replace the venv path:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_app.py
```

Everything else is identical. The codebase is OS-agnostic; the only
Windows-specific assumption is in path examples in the README.

## Cloud — Google Colab (for fine-tuning, not serving)

Colab is the realistic path to >85 % accuracy on this dataset (see
[TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the full recipe). Colab is
**not** a good place to host the demo — its 90-min idle disconnect and
ephemeral filesystem make it awkward for serving — but it's perfect for
training.

To serve the trained model after Colab training, download the resulting
`.pth` file and copy it into `models/` on your local machine, then
`python run_app.py` as above.

## Containerised deployment (optional, Docker)

A minimal Dockerfile (write this to `Dockerfile` at the repo root):

```dockerfile
FROM python:3.11-slim

# OpenCV / facenet-pytorch system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV APP_HOST=0.0.0.0
ENV APP_PORT=5000
EXPOSE 5000

CMD ["python", "run_app.py"]
```

Build and run:
```bash
docker build -t deepfake-detection .
docker run --rm -p 5000:5000 \
  -v "$PWD/models:/app/models:ro" \
  -e APP_SECRET_KEY=$(openssl rand -hex 32) \
  deepfake-detection
```

Note: the image is large (~3 GB) because it bundles PyTorch + CUDA
runtime libraries even on CPU. For a smaller image, switch to a
PyTorch-only base or strip the CUDA libs.

## Production checklist (before exposing this on the internet)

This demo was built for course evaluation, not as a hardened service.
Before exposing it beyond `localhost`:

- [ ] Replace `Config.SECRET_KEY` with a real random secret (≥32 bytes).
      Set via `APP_SECRET_KEY` env var — never commit it.
- [ ] Put a real WSGI server in front (e.g. `gunicorn` on Linux,
      `waitress` on Windows). The `flask run` development server is
      explicitly not for production.
- [ ] Reverse-proxy through nginx / Caddy with HTTPS (Let's Encrypt).
- [ ] Add rate limiting (e.g. `flask-limiter`) — model inference is
      CPU-heavy and easy to DoS.
- [ ] Sandbox uploads — the current code already restricts size and
      extension, deletes files after processing, and uses
      `werkzeug.utils.secure_filename` for the on-disk path. Audit
      these before broader exposure.
- [ ] Constrain CORS — the current app has none configured (browser
      same-origin policy is the only protection).
- [ ] Add structured monitoring of `app/logs/predictions.log` and
      `app/logs/error.log`. Both are file-rotated but not exfiltrated
      anywhere.
- [ ] Containerise (Docker section above) and run as a non-root user.

## Troubleshooting

### "Address already in use" on port 5000

Another process has the port. Either kill it:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process
```
or change the port:
```powershell
$env:APP_PORT = "5050"
F:\ml_project\.venv\Scripts\python.exe run_app.py
```

### "Model checkpoint not found" at startup

The Flask app looks for `models/<kind>_best.pth`. Either retrain (see
[TRAINING_GUIDE.md](TRAINING_GUIDE.md)) or set `APP_MODEL_KIND` to
whichever model file you do have.

### "503 model_not_loaded" from `/predict`

Check `app/logs/error.log` for the underlying load error — usually a
state_dict shape mismatch (you trained with different LSTM hidden size
than `Config.LSTM_HIDDEN_SIZE`) or a corrupt checkpoint.

### Predictions take 60+ seconds

You're hitting CPU thrashing from another running process (browser,
indexer). Close them and retry. Cold start (first prediction) is
slower than subsequent because PyTorch lazy-loads CUDA stubs and
ImageNet means; ~15 s on this laptop is normal.

### Browser shows "413 Request Entity Too Large"

Upload exceeds `Config.MAX_CONTENT_LENGTH` (50 MB by default). Either
reduce the video size (re-encode to ~2-3 Mbps with `ffmpeg`) or raise
the limit in `app/config.py`. The cap exists to protect server RAM
during preprocessing; raising it without raising RAM is risky.

### Predictions seem wildly wrong

The deployed model has 77.58 % overall test accuracy and **41.6 %
recall on real videos** — that means ~6 of every 10 real videos will be
misclassified as fake. This is the model's actual behaviour, not a bug.
See the "Limitations" section of [README.md](README.md).

If predictions are *consistently* wrong (e.g. 100 % FAKE on every
video including the project's known-real test videos), the model
checkpoint may be corrupt — re-download or retrain.
