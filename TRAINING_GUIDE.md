# Training Guide

How to retrain or extend the deepfake detector. Two supported paths:

- **CPU path** (this laptop / similar low-end hardware): cache features
  once, train a small head on cached features. ~2 hr feature pass + ~10 min
  per training run.
- **GPU path** (Google Colab T4 free tier or any 8 GB+ GPU): full
  end-to-end fine-tuning of the ResNet-50 backbone. ~30 min total. Much
  better accuracy.

The deployed model in this repo is the **CNN baseline** (frame-level
ResNet-50 with one trainable block) — see [README.md](README.md) for
final numbers.

## Hardware reality check

| Resource | This laptop (i5-7200U, 8 GB) | Recommended for re-training |
|---|---|---|
| CPU forward pass on ResNet-50 (batch=32) | ~5 s | <0.1 s on T4 |
| End-to-end hybrid training time per epoch (700 videos) | ~4 hours | ~3 min on T4 |
| 1000-video feature cache | ~80 min | ~5 min on T4 |

If you have access to a GPU, use it. The CPU path was built because there
was no GPU available, not because it's preferred.

## CPU path — train the LSTM head on cached features

This is the path the project actually used. It's split across three
scripts so each piece can be re-run independently.

### Step 1 — Cache features

Choose **one** backbone:

#### A) Frozen ImageNet ResNet-50 (general-purpose features)
```powershell
F:\ml_project\.venv\Scripts\python.exe extract_features.py
```
Output: `data/features/<split>/<video_id>.npy`, shape `(32, 2048)`.
Wall time: ~80 min on this laptop.

> ⚠️ ImageNet features did **not** transfer well on this dataset
> (test ROC-AUC 0.44, anti-correlated with truth). Use option B instead.

#### B) Trained CNN baseline as backbone (deepfake-specific features)
```powershell
F:\ml_project\.venv\Scripts\python.exe extract_features_cnn.py
```
Output: `data/features_cnn/<split>/<video_id>.npy`, shape `(32, 2048)`.
Wall time: ~80 min. The CNN baseline must already be trained and saved at
`models/cnn_baseline_best.pth` (it is, in this repo).

Both scripts:
- Skip already-cached files unless `--overwrite` is passed.
- Process train → val → test in that order.
- Print a per-split summary at the end with class counts.

Smoke-test before committing to a full run:
```powershell
F:\ml_project\.venv\Scripts\python.exe extract_features_cnn.py --max-videos 4 --splits val
```

### Step 2 — Train the LSTM head

```powershell
F:\ml_project\.venv\Scripts\python.exe train_hybrid_v2.py `
  --features-dir data/features_cnn `
  --epochs 30 `
  --batch-size 32 `
  --learning-rate 5e-4 `
  --weight-decay 1e-3 `
  --lstm-hidden-size 128 `
  --lstm-num-layers 1 `
  --dropout 0.5 `
  --early-stop-patience 6
```

What it does:
- Loads cached features (no augmentation — pure cached arrays).
- Trains only the BiLSTM head; no backbone gradients.
- Uses `WeightedRandomSampler` to make each batch class-balanced (the
  4:1 fake:real skew otherwise causes a "predict-everything-fake"
  collapse).
- Early-stops on **val macro-F1** (not accuracy — accuracy is misleading
  under class imbalance).
- At the end, plugs the trained head into a fresh `HybridCNNLSTM`
  (with whichever backbone matches the feature cache) and saves the
  full state_dict to `models/hybrid_best.pth`. The existing
  `evaluate_hybrid.py` loads this unchanged.

### Step 3 — Evaluate

Fast, uses cached features:
```powershell
F:\ml_project\.venv\Scripts\python.exe evaluate_hybrid_cached.py --features-dir data/features_cnn
```
Outputs:
- `outputs/hybrid_evaluation.txt` (test report at val-tuned threshold)
- `outputs/hybrid_evaluation_default.txt` (test report at threshold 0.5)
- `outputs/hybrid_threshold_sweep.csv` and `.png`
- `outputs/threshold.json` (tuned threshold the Flask app picks up
  automatically)

## CPU path — retrain the CNN baseline

If you want to retrain the CNN baseline from scratch (e.g. with different
augmentations or a class-balanced sampler):

```powershell
F:\ml_project\.venv\Scripts\python.exe train_cnn.py `
  --epochs 10 `
  --batch-size 16 `
  --learning-rate 5e-5 `
  --trainable-backbone-layers 1 `
  --num-workers 0
```

Note: `train_cnn.py` currently uses plain `BCEWithLogitsLoss` without
class weighting. The recall-on-real collapse the original CNN baseline
showed is fixable in two complementary ways:

1. **Add class weighting / balanced sampler in training** — edit
   `train_cnn.py` to either use `pos_weight=0.25` in `BCEWithLogitsLoss`
   or pass a `WeightedRandomSampler` to the train loader.
2. **Threshold-tune at inference** — already done by
   `tune_cnn_threshold.py` (no retraining needed). The deployed app uses
   threshold 0.75 from this tuner.

## GPU path — fine-tune the hybrid end-to-end

This is the path that hits 90 %+ test accuracy on FaceForensics++ in the
literature.

### Setup on Google Colab

1. New notebook → Runtime → Change runtime type → T4 GPU.
2. Upload the repo (or `git clone` if you push it).
3. Upload the dataset / processed frames (or extract from raw videos
   inside the notebook — slow but free).
4. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
5. Run end-to-end training:
   ```python
   !python train_hybrid.py \
       --epochs 15 \
       --batch-size 4 \
       --num-frames 16 \
       --learning-rate 5e-5 \
       --weight-decay 1e-4 \
       --trainable-backbone-layers 2 \
       --num-workers 2
   ```
   On a T4 with batch=4 and 16 frames per clip you'll fit comfortably in
   16 GB VRAM. Each epoch is ~3 min for 700 train videos.

### Hyperparameter notes

| Knob | Suggestion | Why |
|---|---|---|
| `--learning-rate` | 5e-5 (backbone) | Standard for fine-tuning ImageNet weights — too high a LR catastrophically forgets ImageNet's general features. |
| `--trainable-backbone-layers` | 2 (i.e. `layer3` + `layer4`) | More than 2 risks overfitting on 700 videos. |
| `--num-frames` | 16 (Colab) / 32 (more memory) | 16 frames is enough temporal context and halves memory. |
| `--batch-size` | 4 (T4) / 16 (24 GB GPU) | Use the largest batch size that fits. |
| `--dropout` | 0.5 (LSTM) / 0.4 (classifier) | Keep regularization on; 700 videos is a small set. |
| `--weight-decay` | 1e-4 | Standard. |
| Class balancing | **Add `pos_weight=0.25` in `BCEWithLogitsLoss` or `WeightedRandomSampler`** | The current `train_hybrid.py` does **not** do class weighting. This is the single biggest fix to apply if retraining. |

### Suggested patch to `train_hybrid.py`

The current trainer doesn't class-balance. Apply one of:

```python
# In create_hybrid_model_and_optimizer(), replace:
criterion = nn.BCEWithLogitsLoss()
# with:
pos_weight = torch.tensor([n_real / n_fake], device=device)  # = 0.25
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

or, for a sampler-based fix in `train_hybrid.py`:

```python
from torch.utils.data import WeightedRandomSampler
labels = [lbl for _, lbl in train_dataset.videos]
class_counts = {0: labels.count(0), 1: labels.count(1)}
sample_weights = [1.0 / class_counts[l] for l in labels]
sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          sampler=sampler, num_workers=args.num_workers,
                          pin_memory=torch.cuda.is_available())
```

## Datasets

The split CSVs in `data/splits/` (`train.csv`, `val.csv`, `test.csv`) point
to folders under `data/processed/{real, fake}/...`. Each row is one video,
each folder contains 32 `frame_XX.jpg` face crops at 224×224.

Class distribution (this project's split):

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| train | 140 | 560 | 700 |
| val | 30 | 120 | 150 |
| test | 30 | 120 | 150 |

To regenerate from raw videos:
```powershell
F:\ml_project\.venv\Scripts\python.exe -m src.preprocessing.run_pipeline
```
This takes ~3–4 hours on this laptop and requires the raw FaceForensics++
videos in `data/raw/`. See `src/preprocessing/run_pipeline.py` for the
exact steps.

## Troubleshooting

### Training: val macro-F1 stuck near 0.5

This is the symptom of class collapse — model predicts almost
everything as the majority class. Check:
- Are you using `WeightedRandomSampler` or `pos_weight`? (`train_hybrid_v2.py`
  does the former by default; `train_hybrid.py` does neither — see patch
  above.)
- Is your early-stopping metric val accuracy? Switch to val macro-F1.
- Is the LR too high? Try 5e-4 → 1e-4.

### Training: val accuracy high but val macro-F1 low

Same symptom, viewed differently — the model nails the majority class
and ignores the minority. Same fix as above.

### Training: val loss climbs while train loss drops

Classic overfitting. With 700 train videos this is easy to hit. Reduce
LSTM size (`--lstm-hidden-size 64 --lstm-num-layers 1`), increase
dropout (`--dropout 0.6+`), increase `--weight-decay` (`1e-3` → `1e-2`),
and / or reduce `--early-stop-patience`.

### Training crashes with `RuntimeError: CUDA out of memory`

Reduce `--batch-size` or `--num-frames`. On a T4 with `--num-frames 16`,
batch sizes of 4–6 are typical for the full hybrid.

### Feature extraction is slow or stalls

On low-RAM machines, close the Flask server, browser, and any other
heavy app while extracting. `extract_features*.py` batches 32 frames
through ResNet-50 at a time; if RAM swaps, throughput collapses.

### "ModuleNotFoundError: No module named 'torch'" inside Colab

Colab pre-installs torch in most runtimes, but in some images you may
need:
```python
!pip install torch torchvision
```
```python
import torch; print(torch.__version__, torch.cuda.is_available())
```

### Test ROC-AUC < 0.5

The model is anti-correlated with truth on test — its predictions
inverted would be more accurate than as-is. This means the head
memorised training-set quirks that don't generalise. The fix is
**better features** (use `extract_features_cnn.py` instead of the
ImageNet variant) and / or **GPU end-to-end fine-tuning**, not more
head training.
