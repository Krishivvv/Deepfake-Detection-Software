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

1. **Hugging Face Hub model repo** *(chosen)*
   - Create a model repo: `krishivvv/veridex-deepfake`.
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
# 0. Auth once on THIS machine (weights live here; an Action can't upload them)
huggingface-cli login

# 1. One command does everything: model repo + weights + Space files
python scripts/deploy_space.py

# 2. Set Space variables (https://huggingface.co/spaces/krishivvv/Veridex → Settings):
#    VERIDEX_HF_REPO=krishivvv/veridex-deepfake
#    APP_MODEL_KIND=hybrid_v3
```

The Space builds, pulls weights from the Hub at startup, and serves a public
URL. Put that URL at the top of [README.md](README.md).

## 5. What must NEVER be deployed/committed
- The FaceForensics++ dataset or any `data/` contents (gitignored).
- `.pth` weights in plain git (gitignored; hosted on Hub/LFS instead).
- `app/users.db`, secrets, or `APP_SECRET_KEY` (set real secrets via env).

## 6. Portfolio recommendation

**Pin this repo and position it as your flagship ML/CV project.** It is the
strongest portfolio piece because it spans the full lifecycle: data pipeline →
transfer-learning → temporal modelling → calibrated thresholds → a working
full-stack product → a one-click public demo → tests/CI/model card.

- **GitHub pin order:** #1 of your six pinned repos. Lead with the live-demo
  link and the metrics table — recruiters skim.
- **Resume line:** "Veridex — deepfake video detector (ResNet-50 + BiLSTM,
  PyTorch). 0.80 acc / 0.87 ROC-AUC on FaceForensics++; shipped a Flask +
  Next.js app and a public Gradio demo with Grad-CAM." Keep the name
  **Veridex** (matches this repo; align the resume spelling to it).
- **Talking points for interviews:** train/serve preprocessing parity;
  why a cached-feature LSTM head beats end-to-end on CPU; val-tuned decision
  threshold and the precision/recall trade-off; honest limitations (real-class
  recall, dataset scope).

### Set the repo description + topics
Outward-facing change — run these yourself after review (requires `gh auth login`):

```bash
gh repo edit Krishivvv/Deepfake-Detection-Software \
  --description "Veridex — deepfake video detector (ResNet-50 + BiLSTM, PyTorch). Flask + Next.js app and a Gradio demo with Grad-CAM. 0.80 acc / 0.87 ROC-AUC on FaceForensics++." \
  --add-topic deepfake-detection \
  --add-topic computer-vision \
  --add-topic pytorch \
  --add-topic resnet \
  --add-topic transfer-learning \
  --add-topic lstm \
  --add-topic gradio \
  --add-topic huggingface-spaces
```

---
*This record is the source of truth for the public demo. Update the live-demo
URL in [README.md](README.md) once the Space is live.*
