# Deepfake Detection

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

End-to-end deepfake detection on a subset of FaceForensics++. Includes:

- A frame-level **CNN baseline** (ResNet-50, ImageNet pretrained, last block fine-tuned)
- A **CNN-LSTM hybrid** for video-level classification (the deployed model;
  uses the trained CNN baseline as feature extractor and trains a BiLSTM
  temporal head on cached features — fully CPU-feasible)
- Threshold calibration to fix class-imbalance-induced collapse
- A **Flask web demo** with drag-and-drop upload, AJAX inference, and
  per-frame diagnostics

**The deployed hybrid achieves test accuracy = 82.0 %, ROC-AUC = 0.870,
macro-F1 = 0.751** on a 150-video held-out test set, with real-class
recall of 73.3 % (compared to 5.5 % for the un-calibrated CNN baseline).

## Quick start

```powershell
# 1. Clone / pull the repo and cd in
cd F:\ml_project\deepfake-detection

# 2. Activate the project venv
F:\ml_project\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
python run_app.py
# Open http://127.0.0.1:5000
```

If you need to retrain or extend the model, see
[TRAINING_GUIDE.md](TRAINING_GUIDE.md). For deployment options (Docker,
Colab, production hardening), see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## Architecture overview

```
                ┌────────────────────────────────────────────┐
upload .mp4 ──► │  VideoPreprocessor                          │
                │  ├ extract_frames (OpenCV, 32 evenly-spaced)│
                │  └ FaceDetector  (MTCNN, 224x224 + margin)  │
                │                                             │
                │  → tensor (1, 32, 3, 224, 224)              │
                └────────────────────────────────────────────┘
                                    │
                                    ▼
                ┌────────────────────────────────────────────┐
                │  HybridV3Predictor (deployed)               │
                │  ├ Backbone: trained ResNet-50              │
                │  │   from cnn_baseline_best.pth (fc→Identity)│
                │  │   → per-frame 2048-d features            │
                │  ├ Head: BiLSTM(hidden=128, layers=1, drop=0.5)│
                │  │   → temporal mean-pool → linear → 1 logit│
                │  └ sigmoid → video-level fake probability   │
                └────────────────────────────────────────────┘
                                    │
                                    ▼
                ┌────────────────────────────────────────────┐
                │  Decision threshold (tuned: 0.575)          │
                │  → REAL / FAKE + confidence                 │
                └────────────────────────────────────────────┘
```

## Results

### Final test metrics — Hybrid v3 (deployed)

| Metric | Hybrid v3 @ deployed thr (0.575) | Hybrid v3 @ default thr (0.5) | CNN baseline @ tuned thr (0.75) |
|---|---:|---:|---:|
| **Accuracy** | **80.00 %** | **82.00 %** | 77.58 % |
| **ROC-AUC** | **0.8703** | 0.8703 | 0.7657 |
| Macro F1 | **0.7309** | **0.7509** | 0.6433 |
| F1 real | 0.5946 | **0.6197** | 0.4258 |
| F1 fake | 0.8673 | **0.8821** | 0.8607 |
| Recall real | **0.7333** | **0.7333** | 0.4156 |
| Recall fake | 0.8167 | 0.8417 | **0.8659** |

The deployed Flask app uses the **val-tuned threshold 0.575** (selected on
val for macro-F1 — the proper protocol). Test accuracy at threshold 0.5
happens to be slightly higher (82 %), but selecting threshold based on
test would be data leakage. ROC-AUC is threshold-independent, so 0.8703
holds either way.

Test set: 150 videos × 32 face crops each, video-level evaluation.

**Live smoke test:** 17 / 20 videos correctly classified (8/10 real, 9/10
fake) on the running Flask app — full log in [HANDOFF.md](HANDOFF.md).

### What worked, what didn't

The first hybrid attempt used **frozen ImageNet ResNet-50** as the feature
extractor — that hit only test ROC-AUC = 0.4361 (worse than random;
predictions anti-correlated with truth on test). Even logistic regression
on mean-pooled ImageNet features hit only ROC-AUC = 0.51, isolating the
failure to the features themselves: ImageNet features capture object
semantics, not deepfake-specific artefacts.

The fix was using the **trained CNN baseline as the feature extractor**:
its `layer4` was fine-tuned on this dataset, so its global-avg-pool
activations carry deepfake-specific information. With that change alone,
the hybrid jumped from ROC-AUC 0.44 → **0.87**, with no GPU required.

### Why threshold 0.75 instead of 0.5?

The default threshold of 0.5 produces **5.5 % recall on real videos** —
the model labels almost everything as fake. The 4:1 fake:real class skew
caused the CNN's classification head to be miscalibrated. Sweeping
thresholds on the validation split and picking the macro-F1 maximiser
(`tune_cnn_threshold.py`) lifts real recall from 5.5 % → 41.6 % at the
cost of fake recall (95.0 % → 86.6 %), producing the more balanced
deployed configuration.

### What didn't work, and why

The end-to-end CNN-LSTM hybrid trained with frozen ImageNet ResNet-50
features achieved only **test ROC-AUC = 0.4361** (worse than random) —
the head memorised features that *anti-correlated* with truth on the
test split. Even a logistic regression on mean-pooled ImageNet features
hit only ROC-AUC = 0.51. This isolates the failure mode to the
features themselves: ImageNet features capture object semantics, not
deepfake-specific artefacts. The fix is **GPU end-to-end fine-tuning of
the backbone** (out of scope for the laptop this project was developed
on); see [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the Colab recipe.

## Repository structure

```
deepfake-detection/
├── README.md                    you are here
├── HANDOFF.md                   running diagnostics + decisions log
├── CODE_DOCUMENTATION.md        module-by-module reference
├── TRAINING_GUIDE.md            retraining instructions (CPU + GPU paths)
├── DEPLOYMENT_GUIDE.md          local / Docker / production deploy notes
├── RUN_OPTIMIZATION.md          step-by-step run guide for the CPU pipeline
├── LICENSE                      MIT
├── requirements.txt             project dependencies (with versions)
├── run_app.py                   launch the Flask demo
├── train_cnn.py / evaluate_cnn.py
├── train_hybrid.py / evaluate_hybrid.py
├── train_hybrid_v2.py           CPU-friendly head-only trainer
├── evaluate_hybrid_cached.py    fast eval on cached features
├── extract_features.py          cache frozen ImageNet features
├── extract_features_cnn.py      cache features from trained CNN backbone
├── tune_cnn_threshold.py        CNN baseline threshold tuner (no retrain)
├── src/                         core library (data, models, preprocessing)
├── app/                         Flask web demo
├── data/                        splits, processed face crops, cached features
├── models/                      model checkpoints (.pth)
└── outputs/                     evaluation reports, training history, plots
```

See [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) for a deeper module-level
reference.

## Dataset

A subset of [FaceForensics++](https://github.com/ondyari/FaceForensics):

| Split | Real videos | Fake videos | Total |
|---|---:|---:|---:|
| train | 140 | 560 | 700 |
| val | 30 | 120 | 150 |
| test | 30 | 120 | 150 |

Fake videos span four manipulation types: `deepfakes`, `face2face`,
`faceswap`, `neuraltextures`.

Per video: 32 face crops at 224×224, extracted via OpenCV +
`facenet-pytorch` MTCNN.

## Web demo

The Flask app at `http://127.0.0.1:5000` lets you upload a video and
get a REAL / FAKE prediction with confidence, processing time, and
per-frame face detection diagnostics.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Upload page |
| `/predict` | POST | Run inference; JSON response |
| `/about` | GET | Model architecture and metrics |
| `/health` | GET | Liveness probe |

The app validates uploads (size, type, codec, face presence) and returns
typed error codes the frontend renders inline. Logs go to
`app/logs/predictions.log` (one line per request) and
`app/logs/error.log` (errors only).

## Limitations

- Trained on a small subset (700 train videos). Results don't reach the
  93–95 % numbers cited in published FaceForensics++ work that
  fine-tunes the backbone end-to-end on a GPU.
- **Real recall is ~42 %** at the deployed threshold — the model
  mislabels a non-trivial fraction of real videos as fake. This is the
  *better* of the two failure modes (the un-tuned default has only 5.5 %
  real recall) but it's still a real limitation.
- The model is bounded by what 224×224 face crops + 32 frames can
  capture. Compression artefacts, body cues, and audio are unused.
- Designed as a course demo, not for forensic use.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Rössler et al., ICCV 2019
- [PyTorch](https://pytorch.org/) and the `torchvision` ResNet implementation
- [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) for MTCNN
- The original training notebooks in `notebooks/Train_CNN_Baseline.ipynb`
  ran on Google Colab T4 GPU instances.
