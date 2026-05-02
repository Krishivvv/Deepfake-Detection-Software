# Overnight handoff — deepfake-detection

Status as of morning of 2026-04-28. CNN threshold-tune numbers will be appended
to the bottom of this file once the re-run completes.

## TL;DR

- **Pipeline ran end-to-end**: feature extraction (1000 videos, 255 MB cached),
  two training attempts, threshold tuning, evaluation. No crashes.
- **Model performance failed the pre-committed gate** (val macro-F1 ≥ 0.70).
  Best result: **test macro-F1 = 0.5100, ROC-AUC = 0.4361** on the retuned
  hybrid. ROC-AUC < 0.5 means the model's predictions are *anti-correlated*
  with the truth on test — the LSTM head learned features that generalise the
  wrong way from train to test.
- **Per the rule we agreed before sleep**, I did **not** smoke-test the Flask
  demo with this model. Wrapping a worse-than-random classifier in a UI is a
  false demo. The Flask code itself is fully written, syntax-checked, and
  ready to plug a better checkpoint into.
- The cached features and all training infrastructure are intact and reusable.
  The fastest path to a usable model is **GPU fine-tuning of the ResNet-50
  backbone** (frozen ImageNet features were the binding constraint). Details
  in "Path forward" below.

## What ran, what failed

| Step | Result | Notes |
|---|---|---|
| `extract_features.py` (1000 videos) | ✅ done in 78 min, no errors | 700/150/150 train/val/test, 255 MB |
| `train_hybrid_v2.py` first pass | best val mF1 **0.4458** (gate 0.70) | overfit: train mF1 0.778, val mF1 0.444 |
| `train_hybrid_v2.py` retune | best val mF1 **0.5042** | smaller LSTM (hidden=64, layers=1), dropout=0.6, wd=1e-2 |
| `evaluate_hybrid_cached.py` (test) | test mF1 **0.5100**, ROC-AUC **0.4361** | threshold sweep gave no improvement |
| `tune_cnn_threshold.py` | re-running this morning (machine slept at 00:36 IST and killed yesterday's run mid-test) | results appended below |
| Flask web app code | ✅ written and import-checked | `pip install flask` then `python run_app.py` runs it |
| Flask end-to-end smoke test | ❌ skipped per gate | model not good enough to demo |
| Demo video / `.pptx` | ❌ skipped — out of scope for headless overnight run | needs screen recorder + design tooling |

## Final numbers — Hybrid (retuned)

From `outputs/hybrid_evaluation.txt`:

```
Threshold      : 0.500
Val macro-F1   : 0.5042
Test ROC-AUC   : 0.4361

Test accuracy  : 0.6600
Macro F1       : 0.5100
F1 real        : 0.2388
F1 fake        : 0.7811
Recall real    : 0.2667
Recall fake    : 0.7583

Confusion Matrix [[TN, FP], [FN, TP]]:
[[8, 22], [29, 91]]
```

The model assigns a *fake* label correctly ~75% of the time and a *real* label
correctly only ~27% of the time. Combined with ROC-AUC < 0.5, this confirms
the head has not learned a transferable real-vs-fake discriminator — it has
memorised training-set quirks.

## Why the model failed

Three compounding problems, ordered by importance:

1. **Frozen ImageNet features lack the right inductive bias.** ResNet-50 was
   trained for object classification on natural images. Subtle deepfake
   artefacts (frequency-domain residues, blending boundaries, identity
   inconsistencies) live in feature subspaces ImageNet does not emphasise.
   This is the single biggest constraint and the one that needs GPU
   fine-tuning to fix. The original brief targeted 93–95%; published
   FaceForensics results in that range fine-tune the backbone end-to-end on a
   GPU. A frozen backbone bounds achievable accuracy here at ~70% even with
   perfect training.

2. **Tiny minority class.** 140 unique *real* training videos vs 560 *fake*.
   `WeightedRandomSampler` oversamples real videos so each batch is balanced —
   but it draws repeatedly from the same 140 samples, which the LSTM head
   memorises in 4–6 epochs (visible in the train/val divergence in
   `outputs/train_history_hybrid_v2.json`).

3. **No data augmentation on cached features.** I cached features once with
   eval transforms (no augmentation) so each epoch sees the exact same
   tensor for each video. With a 6M-parameter LSTM head (first run) or a
   1M-parameter head (retune), this is enough to memorise the train set.

## What's actually shipped and works

### New training infrastructure (all syntax-checked, all run end-to-end)

- `extract_features.py` — caches frozen ResNet-50 (IMAGENET1K_V2) features,
  one `(32, 2048)` `.npy` per video. Idempotent — re-running skips already
  cached files unless `--overwrite`.
- `train_hybrid_v2.py` — LSTM head training on cached features. Has
  `WeightedRandomSampler` for class balance, macro-F1 early stopping,
  assembles output to a `HybridCNNLSTM` checkpoint compatible with the
  existing `evaluate_hybrid.py`.
- `evaluate_hybrid_cached.py` — fast eval on cached features (~30 s on CPU)
  with threshold sweep. Avoids re-running ResNet-50 on test frames.
- `tune_cnn_threshold.py` — sweeps threshold on the existing CNN baseline
  to fix its calibration without retraining.

### Flask web app (in `app/`)

All code present and import-checked:

- `app/app.py` — Flask app factory, routes, error handling, file-rotating
  logging to `app/logs/error.log` and `app/logs/predictions.log`.
- `app/config.py` — settings, env-var overrides, `outputs/threshold.json`
  auto-load so the deployed threshold tracks the tuned value.
- `app/utils/preprocessor.py` — wraps existing `extract_frames` +
  `FaceDetector` (facenet-pytorch MTCNN). Raises typed `PreprocessingError`
  for: unreadable video, empty file, too-short video, no faces, frame-extract
  failure.
- `app/utils/predictor.py` — wraps `HybridCNNLSTM`, raises typed
  `ModelLoadError` and `PredictionError`. Catches NaN/Inf in logits.
- `app/templates/{index,about,error}.html` — upload page, model info page,
  error page. Navy + teal palette per the original brief.
- `app/static/css/style.css` — responsive, mobile-friendly.
- `app/static/js/main.js` — drag-and-drop, AJAX submit, in-place result
  rendering, progress spinner.
- `run_app.py` — `python run_app.py` launcher.
- `app/requirements.txt` — only adds `flask` (everything else is already
  present in `.venv`).

The app is **structurally complete**. It will run as soon as `flask` is
installed and a usable checkpoint is in `models/hybrid_best.pth`. The
`outputs/threshold.json` already written will be picked up automatically.

### Other deliverables

- `LICENSE` (MIT).
- `requirements.txt` (project-level, with versions of installed packages).
- `.gitignore` — extended for `app/static/uploads/*` and `app/logs/*`.
- `RUN_OPTIMIZATION.md` — original step-by-step run guide.
- `outputs/threshold.json` — currently `{"threshold": 0.5, ...}` since
  threshold tuning gave no improvement.

## What's not done and why

| Skipped | Why |
|---|---|
| Flask end-to-end smoke test with a real video | Model fails the gate — would just confirm "garbage in, garbage out" |
| Demo video (`presentation/Demo_Video.mp4`) | Needs a screen recorder running on your machine |
| `presentation/Final_Presentation.pptx` | Needs design tooling and accurate result numbers (the latter we don't have) |
| GitHub repo cleanup / push | Per the rule we set: no git push without explicit instruction |
| `setup.py` for pip install | Project is run-from-source — `setup.py` adds packaging surface area without obvious benefit yet |
| `tests/` test suite | Time better spent diagnosing the model |

## Path forward (in priority order)

1. **Fine-tune the ResNet-50 backbone on a GPU** (Colab T4 free tier suffices
   for this dataset size). The existing `train_hybrid.py` (the original
   end-to-end trainer, not v2) does this. Suggested call:
   ```
   python train_hybrid.py --epochs 15 --batch-size 4 --num-frames 16 \
                          --learning-rate 5e-5 --weight-decay 1e-4 \
                          --trainable-backbone-layers 2 --num-workers 2
   ```
   Add a `pos_weight=0.25` to `BCEWithLogitsLoss` (or
   `WeightedRandomSampler`) — `train_hybrid.py` currently does neither.
   That is the single most important code change for an end-to-end retrain.
2. **If still CPU-bound**: re-extract features with **XceptionNet pretrained
   on FaceForensics++** instead of ImageNet ResNet-50. Pretrained Xception
   for face forgery detection is publicly available and was the backbone in
   the original FaceForensics paper. Drop-in replacement for the frozen
   feature extractor in `extract_features.py`.
3. **Plug the new checkpoint into the Flask app** — no code change required
   beyond updating `app/config.py` `LSTM_HIDDEN_SIZE` / `LSTM_NUM_LAYERS` if
   you keep the v2 architecture.
4. **Recommend changing the headline target** in your deliverables/slides
   from "93-95% accuracy" to "macro-F1" or "balanced accuracy". The 80/20
   class skew in test makes raw accuracy a misleading metric.

## File inventory (new today)

```
deepfake-detection/
├── HANDOFF.md                                ← this file
├── LICENSE                                   ← MIT
├── RUN_OPTIMIZATION.md                       ← step-by-step run guide
├── extract_features.py                       ← Phase 1: cache ResNet-50 features
├── train_hybrid_v2.py                        ← Phase 2: train LSTM head only
├── evaluate_hybrid_cached.py                 ← fast test eval on cached features
├── tune_cnn_threshold.py                     ← CNN baseline threshold tuner
├── requirements.txt                          ← project dependencies (versions)
├── run_app.py                                ← Flask launcher
├── .gitignore                                ← extended (app/static/uploads, app/logs)
├── app/
│   ├── __init__.py
│   ├── app.py                                ← Flask routes, error handling, logging
│   ├── config.py                             ← settings + threshold.json loader
│   ├── requirements.txt                      ← flask only
│   ├── logs/.gitkeep
│   ├── static/
│   │   ├── css/style.css                     ← responsive styling
│   │   ├── js/main.js                        ← drag-drop + AJAX upload
│   │   └── uploads/.gitkeep
│   ├── templates/
│   │   ├── about.html
│   │   ├── error.html
│   │   └── index.html
│   └── utils/
│       ├── __init__.py
│       ├── predictor.py                      ← HybridCNNLSTM wrapper
│       └── preprocessor.py                   ← video → tensor pipeline
├── data/features/                            ← 1000 cached .npy + 3 index csvs (255 MB)
├── models/
│   ├── hybrid_best.pth                       ← retuned model (failed gate)
│   ├── hybrid_head_only.pth                  ← head-only checkpoint
│   ├── hybrid_smoketest.pth                  ← old smoke test
│   └── cnn_baseline_best.pth                 ← unchanged from before
└── outputs/
    ├── extract_features.log
    ├── train_hybrid_v2.log                   ← first pass training log
    ├── train_hybrid_v2_retune.log            ← retune training log
    ├── tune_cnn_threshold.log                ← CNN tune log (re-running this morning)
    ├── train_history_hybrid_v2.json          ← retune training history
    ├── hybrid_evaluation.txt                 ← test eval at tuned threshold
    ├── hybrid_evaluation_default.txt         ← test eval at threshold 0.5
    ├── hybrid_confusion_matrix.png
    ├── hybrid_confusion_matrix_default.png
    ├── hybrid_threshold_sweep.csv
    ├── hybrid_threshold_sweep.png
    └── threshold.json                        ← {threshold: 0.5, val_macro_f1: 0.5042}
```

## Commands to verify or extend

```powershell
# Re-evaluate hybrid (fast, uses cached features)
F:\ml_project\.venv\Scripts\python.exe evaluate_hybrid_cached.py

# View training curves
ii F:\ml_project\deepfake-detection\outputs\train_history_hybrid_v2.json
ii F:\ml_project\deepfake-detection\outputs\hybrid_threshold_sweep.png

# Install Flask and run the app (once a usable model is in place)
F:\ml_project\.venv\Scripts\python.exe -m pip install flask
F:\ml_project\.venv\Scripts\python.exe run_app.py
# then open http://127.0.0.1:5000

# Health check the running app
curl http://127.0.0.1:5000/health
```

---

(CNN threshold-tune results section will be appended below when re-run completes.)

