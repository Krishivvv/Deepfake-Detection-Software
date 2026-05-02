# CPU-only retraining plan

The original `hybrid_smoketest.pth` was a 1-epoch sanity model that learned
nothing useful (val ROC-AUC 0.45, predicts everything as "fake"). End-to-end
retraining of ResNet-50 + BiLSTM on CPU is not realistic. This plan splits
the work into a one-time feature extraction pass and a fast head-only
training loop.

All commands assume the project venv:
- Python: `F:\ml_project\.venv\Scripts\python.exe`
- Working dir: `F:\ml_project\deepfake-detection`

## Step 1 — Extract frozen ResNet-50 features (one-time, ~1.5–2 hours)

```powershell
F:\ml_project\.venv\Scripts\python.exe extract_features.py
```

What it does:
- Loads ImageNet-pretrained ResNet-50 (fc replaced by Identity) on CPU.
- For every video in `data/splits/{train,val,test}.csv`, loads its 32 frames,
  runs them through the backbone, and saves a `(32, 2048)` float32 array to
  `data/features/<split>/<video_id>.npy`.
- Writes an index csv per split with `video_id, label, n_frames`.

Expected runtime on this machine: ~6 s/video × 1000 videos ≈ **100 minutes**.
You can run it overnight.

Disk footprint: 1000 × 32 × 2048 × 4 B ≈ **260 MB** total.

Smoke test first (recommended — finishes in ~15 s):
```powershell
F:\ml_project\.venv\Scripts\python.exe extract_features.py --max-videos 4 --splits val
```

If you need to resume after an interruption: rerun the same command. Files
that already exist are skipped unless you pass `--overwrite`.

## Step 2 — Train the LSTM head (~15–25 min on CPU)

```powershell
F:\ml_project\.venv\Scripts\python.exe train_hybrid_v2.py --epochs 30 --batch-size 32 --learning-rate 1e-3
```

What it does:
- Loads the cached features (no augmentation — pure cached arrays).
- Trains only the BiLSTM temporal head (`LSTMTemporalClassifier`,
  `feature_dim=2048, hidden=256, layers=2, bidir, dropout=0.5`).
- Uses `WeightedRandomSampler` so each batch is class-balanced
  (this is the fix for the "always predicts fake" collapse).
- Early-stops on **val macro-F1**, not val accuracy, so a majority-class
  collapse is penalised instead of rewarded.
- At the end, assembles a fresh ResNet-50 (ImageNet IMAGENET1K_V2) +
  the trained LSTM head into a `HybridCNNLSTM` and saves the full
  `state_dict` to `models/hybrid_best.pth`. This file is drop-in compatible
  with `evaluate_hybrid.py` — no evaluator changes needed.

Hyperparameter notes:
- Default LR `1e-3` is high because we are training only a small head;
  LR `1e-4` (the old end-to-end default) trains too slowly here.
- If macro-F1 plateaus, try `--learning-rate 5e-4 --weight-decay 5e-4`
  for stronger regularisation.

## Step 3 — Evaluate the hybrid on test

The existing evaluator works unchanged:
```powershell
F:\ml_project\.venv\Scripts\python.exe evaluate_hybrid.py
```

It will load `models/hybrid_best.pth` and write
`outputs/hybrid_evaluation.txt` and `outputs/hybrid_confusion_matrix.png`
with real numbers.

If at threshold 0.5 the model still leans heavily toward one class, sweep
thresholds:
```powershell
F:\ml_project\.venv\Scripts\python.exe evaluate_hybrid.py --threshold 0.4
F:\ml_project\.venv\Scripts\python.exe evaluate_hybrid.py --threshold 0.6
```

## Step 4 (parallel-able) — Threshold-tune the CNN baseline (no retrain)

This fixes the CNN baseline's recall-on-real collapse without touching
training. Sweeps thresholds on val, picks the macro-F1 maximiser, evaluates
test at that threshold.

```powershell
F:\ml_project\.venv\Scripts\python.exe tune_cnn_threshold.py
```

Outputs:
- `outputs/cnn_evaluation_tuned.txt`
- `outputs/cnn_confusion_matrix_tuned.png`
- `outputs/cnn_threshold_sweep.csv` and `.png`

Expected runtime: ~10–20 min on CPU (one forward pass over val + test
frame-level images).

## Realistic expectations

With **frozen** ImageNet ResNet-50 features (no fine-tuning), 700 train
videos, and CPU-only:

- Hybrid v2 macro-F1 on val: typical range **0.75–0.90**.
- Hybrid v2 accuracy on test: typical range **0.78–0.90**.
- The 93–95% number from the original brief assumes GPU fine-tuning of the
  backbone. Hitting that on CPU here is not realistic.

If results are unsatisfactory after Step 3, options (in order of effort):
1. Tune dropout / weight-decay / LR on the head only (cheap, minutes).
2. Re-extract features with a smaller backbone (ResNet-18 → 512-dim,
   ~6× faster extraction, sometimes better generalisation on small data).
3. Fine-tune the last ResNet-50 stage on Google Colab T4 with the existing
   `train_hybrid.py` (out of scope for the local-CPU plan).

## What this plan does NOT do

- It does not retrain the CNN baseline. The existing
  `cnn_baseline_best.pth` stays untouched. Only its decision threshold is
  tuned.
- It does not regenerate `train_history_local.json` or the original
  `hybrid_training_curves.png`. New training history is written to
  `outputs/train_history_hybrid_v2.json`.
