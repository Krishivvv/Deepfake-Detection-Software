---
title: Veridex Deepfake Detector
emoji: 🛡️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: gradio_app.py
pinned: false
license: mit
short_description: Upload a video; ResNet-50 + BiLSTM flags deepfakes with Grad-CAM.
---

# Veridex — Deepfake Video Detector (live demo)

This Space is the public, CPU-only inference demo for **Veridex**. Upload a
short clip with a visible face and get a **REAL / FAKE** verdict, confidence,
and a Grad-CAM heatmap.

- Model: ResNet-50 backbone (fine-tuned on FaceForensics++) + BiLSTM temporal
  head (`hybrid_v3`). Test accuracy **0.80**, ROC-AUC **0.87**.
- Preprocessing matches training exactly (MTCNN face crop + ImageNet
  normalization, 32 frames @ 224×224).
- **Not for forensic use** — predictions are probabilistic.

## How weights are loaded
This repo does **not** contain model weights. At startup the app pulls them
from a Hugging Face Hub model repo. Set these Space variables:

| Variable | Example |
|---|---|
| `VERIDEX_HF_REPO` | `Krishivvv/veridex-deepfake` |
| `VERIDEX_BACKBONE_FILE` | `cnn_baseline_best.pth` (default) |
| `VERIDEX_HEAD_FILE` | `hybrid_v3_head.pth` (default) |

Full source & model card:
<https://github.com/Krishivvv/Deepfake-Detection-Software>
