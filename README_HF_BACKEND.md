---
title: Veridex API
emoji: 🛡️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Veridex Flask API (ResNet-50 + BiLSTM deepfake model)
---

# Veridex — API backend

This Space is the **Flask + model API** behind the Veridex Next.js frontend
(hosted on Vercel). It is not a UI — it exposes JSON endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | `multipart/form-data video=<file>` → prediction |
| POST | `/api/auth/signup` · `/api/auth/login` | JWT auth |
| GET | `/api/auth/me` | current user |
| GET | `/health` | liveness + `model_loaded` |

Model weights are pulled from the public Hub repo `krishivvv/veridex-deepfake`
at container startup (never committed to git). Configure via Space variables:

| Variable | Value |
|---|---|
| `APP_CORS_ORIGINS` | your Vercel URL, e.g. `https://veridex.vercel.app` (or `*` for any) |
| `APP_MODEL_KIND` | `hybrid_v3` |
| `APP_SECRET_KEY` | a long random string (set as a **secret**) |
| `VERIDEX_HF_REPO` | `krishivvv/veridex-deepfake` |

Source & frontend: <https://github.com/Krishivvv/Deepfake-Detection-Software>
