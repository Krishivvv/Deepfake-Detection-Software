# Veridex — Deployment Decision Record & Runbook

This document records **what we deploy, where, and why**, plus the runbook to
reproduce it. It complements (and supersedes for the public demo)
[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md), which covers the full local
Flask + Next.js stack.

## 1. Decision: deploy inference only

| Question | Decision | Rationale |
|---|---|---|
| Deploy training? | **No.** | Training needs the FaceForensics++ dataset (licensed, ~GBs) and a GPU. Neither belongs in a public demo. Training stays local/Colab (see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)). |
| Deploy a demo? | **Yes — inference only.** | A recruiter/visitor should click a link and get a prediction in seconds. |
| Target platform | **Hugging Face Spaces (Gradio SDK)** | Free CPU tier, zero-ops, public URL, first-class PyTorch support, persistent. |
| Hardware | **CPU is sufficient.** | `hybrid_v3` runs in a few seconds/clip on CPU (the production app already serves on CPU). No GPU cost. |
| Which model | **`hybrid_v3`** (CNN-baseline backbone + LSTM head). | Best test numbers: acc 0.800 / ROC-AUC 0.870 @ thr 0.575. |
| Where do weights live | **Hugging Face Hub model repo** (NOT plain git). | Weights are 100+ MB; git history must stay lean. Pulled at Space startup. Git LFS is the fallback if Hub is undesired. |

### Why not keep only the full-stack app?
The Flask + Next.js app is excellent for a local/portfolio walkthrough but
needs **two processes, a build step, and a backend host** — too much friction
for a "click and try" public link. So we **keep the full-stack app** (unchanged)
*and* add a **single-file Gradio Space** as the public demo. Best of both.

## 2. Weight hosting — the rule

> **Never commit `.pth` files to plain git.** They are already gitignored
> (`*.pth` in `.gitignore`; verified with `git check-ignore`).

Two supported hosting options (pick one at CHECKPOINT B):

1. **Hugging Face Hub model repo** *(recommended)*
   - Create a model repo, e.g. `Krishivvv/veridex-deepfake`.
   - Upload `cnn_baseline_best.pth` (backbone) and `hybrid_v3_head.pth` (head).
   - The Space downloads them at startup via `huggingface_hub.hf_hub_download`,
     controlled by env vars `VERIDEX_HF_REPO`, `VERIDEX_BACKBONE_FILE`,
     `VERIDEX_HEAD_FILE`.
2. **Git LFS on the Space repo**
   - `git lfs track "*.pth"` inside the Space repo only, commit the two files.
   - Simpler, but couples weights to the Space repo.

The two files needed by the demo (head is small):

| File | Size | Role |
|---|---|---|
| `cnn_baseline_best.pth` | ~214 MB | ResNet-50 backbone (fc → Identity at inference) |
| `hybrid_v3_head.pth` | ~9 MB | Trained LSTM temporal head |

## 3. The Space

- Entry point: [`gradio_app.py`](gradio_app.py) — single file, self-contained.
- SDK config: [`README_HF_SPACE.md`](README_HF_SPACE.md) (the Space's own README
  with YAML front-matter Spaces requires).
- Dependencies: [`requirements-space.txt`](requirements-space.txt) (CPU torch,
  gradio, opencv-headless, facenet-pytorch, huggingface_hub).
- Optional container parity: [`Dockerfile.space`](Dockerfile.space).
- Flow: **upload video → exact training preprocessing (MTCNN crop + ImageNet
  normalize) → `hybrid_v3` → REAL/FAKE + confidence + Grad-CAM heatmap.**
- Preprocessing parity is guaranteed because the Space reuses
  `app/utils/preprocessor.py` and `app/utils/predictor.py` — the same code the
  trained model and the Flask backend use.

## 4. Runbook (executed only after CHECKPOINT B sign-off)

```bash
# 0. Auth once
huggingface-cli login

# 1. Host weights (Option 1: Hub model repo)
huggingface-cli repo create veridex-deepfake --type model
huggingface-cli upload Krishivvv/veridex-deepfake models/cnn_baseline_best.pth
huggingface-cli upload Krishivvv/veridex-deepfake models/hybrid_v3_head.pth

# 2. Create the Space (Gradio SDK) and push the demo files
huggingface-cli repo create veridex --type space --space_sdk gradio
#   push: gradio_app.py, requirements-space.txt, README_HF_SPACE.md,
#         src/, app/utils/, app/config.py   (NO weights, NO dataset)

# 3. Set Space env vars (Settings → Variables):
#    VERIDEX_HF_REPO=Krishivvv/veridex-deepfake
#    APP_MODEL_KIND=hybrid_v3
```

The Space builds, pulls weights from the Hub at startup, and serves a public
URL. Put that URL at the top of [README.md](README.md).

## 5. What must NEVER be deployed/committed
- The FaceForensics++ dataset or any `data/` contents (gitignored).
- `.pth` weights in plain git (gitignored; hosted on Hub/LFS instead).
- `app/users.db`, secrets, or `APP_SECRET_KEY` (set real secrets via env).

---
*Portfolio positioning guidance is in [§ Portfolio recommendation](#6-portfolio-recommendation) below.*
