# Backend Dockerfile — used to deploy the Flask + ML service to
# Hugging Face Spaces (Docker SDK) or any other Docker-friendly host.
#
# Build locally:
#     docker build -t veridex-backend .
#     docker run --rm -p 7860:7860 -e APP_CORS_ORIGINS=http://localhost:3000 veridex-backend
#     curl http://localhost:7860/health

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV / facenet-pytorch / FFmpeg system deps. We install the slim set
# needed for headless OpenCV + video decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (smaller, no CUDA) so the requirements step
# below sees torch already satisfied and won't pull the multi-GB CUDA build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# Install the rest of the Python deps (+ huggingface_hub for weight fetch).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt "huggingface_hub>=0.23,<1.0"

# Copy only the parts of the project the runtime needs. Weights are NOT baked
# in — they are pulled from the HF Hub at startup (see prefetch_weights.py).
# Heavy data dirs, notebooks and the frontend are excluded via .dockerignore.
COPY app/ ./app/
COPY src/ ./src/
COPY scripts/prefetch_weights.py ./scripts/prefetch_weights.py
COPY run_app.py ./run_app.py

# HF Spaces sets HOME=/app and runs as a non-root user; ensure the HF cache
# and a writable models/ dir exist.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p models /app/.cache/huggingface

# Hugging Face Spaces (Docker SDK) routes traffic to port 7860 by default.
ENV APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    APP_MODEL_KIND=hybrid_v3 \
    SERVE_WITH_WAITRESS=1 \
    VERIDEX_HF_REPO=krishivvv/veridex-deepfake

EXPOSE 7860

# HF Spaces / Render / Cloud Run override APP_CORS_ORIGINS at deploy time —
# set it to the public Vercel URL of the frontend.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl --fail http://localhost:${APP_PORT}/health || exit 1

# Pull weights from the Hub, then start the production WSGI server.
CMD ["sh", "-c", "python scripts/prefetch_weights.py && python run_app.py"]
