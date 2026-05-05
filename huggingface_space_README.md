---
title: Veridex Backend
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Deepfake video detection — Flask + ResNet-50 + BiLSTM
---

# Veridex Backend

Inference + auth API for the Veridex deepfake detector.

This Space runs the Flask backend behind `waitress`. The model is a hybrid
**ResNet-50 + BiLSTM** trained on FaceForensics++ that achieves
**82.0 % test accuracy / ROC-AUC 0.870** on the held-out test set.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness probe (`{ok, model_loaded}`) |
| POST | `/api/auth/signup` | Create account, returns JWT |
| POST | `/api/auth/login` | Sign in, returns JWT |
| GET | `/api/auth/me` | Current user (Bearer token) |
| POST | `/predict` | Multipart `video` upload → REAL/FAKE prediction |

## Settings to configure on first deploy

In the Space's **Settings → Variables and secrets**, add:

| Variable | Value |
|---|---|
| `APP_CORS_ORIGINS` | the public URL of your Vercel frontend (comma-separated if more than one) |
| `APP_SECRET_KEY` | a random ≥32-character string — used to sign JWTs |

The frontend lives elsewhere (Vercel). Source code: see the project repo.
