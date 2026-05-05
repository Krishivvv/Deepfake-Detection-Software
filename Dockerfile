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

# Install Python deps first so they cache when only source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the parts of the project the runtime needs. Heavy data
# directories (data/raw, data/processed, data/features_cnn, training
# notebooks, the frontend) are excluded via .dockerignore.
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY outputs/threshold_cnn.json outputs/threshold_hybrid_v3.json ./outputs/
COPY run_app.py ./run_app.py

# Hugging Face Spaces (Docker SDK) routes traffic to port 7860 by
# default. Other PaaS hosts will set $PORT — we honour that too.
ENV APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    APP_MODEL_KIND=hybrid_v3 \
    SERVE_WITH_WAITRESS=1

EXPOSE 7860

# HF Spaces / Render / Cloud Run all override APP_CORS_ORIGINS at deploy
# time — set yours to the public Vercel URL of the frontend.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail http://localhost:${APP_PORT}/health || exit 1

CMD ["python", "run_app.py"]
