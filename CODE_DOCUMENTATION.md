# Code Documentation

Module-by-module reference for `deepfake-detection`. Pair this with
[README.md](README.md) for setup, [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for
retraining instructions, and [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for
running the demo.

## Repository layout

```
deepfake-detection/
├── src/                              core library
│   ├── data/
│   │   ├── dataset.py                frame-level dataset
│   │   └── video_dataset.py          video-level (sequence) dataset
│   ├── models/
│   │   ├── resnet_classifier.py      DeepfakeClassifier (CNN baseline)
│   │   ├── lstm_temporal.py          LSTMTemporalClassifier (head)
│   │   └── hybrid_model.py           HybridCNNLSTM (end-to-end)
│   └── preprocessing/
│       ├── extract_frames.py         video → 32 evenly-spaced BGR frames
│       ├── detect_faces.py           MTCNN face detector + cropper
│       ├── create_splits.py          builds data/splits/*.csv
│       └── run_pipeline.py           one-shot raw→processed pipeline
├── train_cnn.py                      Week 3: train CNN baseline
├── evaluate_cnn.py                   evaluate CNN baseline
├── train_hybrid.py                   Week 4: train end-to-end HybridCNNLSTM
├── evaluate_hybrid.py                evaluate HybridCNNLSTM
├── extract_features.py               cache frozen ResNet-50 features (Week 5)
├── extract_features_cnn.py           cache features from trained CNN backbone
├── train_hybrid_v2.py                train LSTM head on cached features
├── evaluate_hybrid_cached.py         fast eval on cached features
├── tune_cnn_threshold.py             threshold-tune CNN baseline (no retrain)
├── run_app.py                        launch Flask demo
└── app/                              Flask web demo
    ├── app.py                        routes, error handling, logging
    ├── config.py                     central settings
    ├── utils/
    │   ├── preprocessor.py           video → tensor pipeline
    │   └── predictor.py              CNN / Hybrid model wrappers
    ├── templates/                    Jinja2 HTML
    └── static/                       CSS, JS, uploads, logs
```

## Core library (`src/`)

### `src/data/dataset.py`

Frame-level PyTorch `Dataset` and DataLoader builder for the CNN baseline.

**`DeepfakeDataset(csv_path, project_root, split, image_size=224, ...)`**
- Reads a split CSV (`train.csv` / `val.csv` / `test.csv`).
- For each row whose `video_path` points to a folder, expands to one sample
  per JPEG inside that folder. For rows pointing to a file, uses the file
  directly.
- Default transforms: `Resize → ToTensor → Normalize(IMAGENET_*)`. The
  training split also applies `RandomHorizontalFlip + RandomRotation(10°) +
  ColorJitter`.
- Returns `(image_tensor, label_tensor)`.

**`build_dataloaders(project_root, batch_size, num_workers, image_size, ...)`**
- Convenience: build train/val/test DataLoaders in one call.
- `num_workers > 0` is fine on Linux/macOS; on Windows prefer `0` to avoid
  the multiprocessing-spawn overhead.

### `src/data/video_dataset.py`

Video-level (sequence) PyTorch `Dataset`.

**`VideoSequenceDataset(csv_path, project_root, split, num_frames=32, image_size=224, ...)`**
- Each row is one video folder; loaded as `(T=32, 3, H, W)`.
- `_select_indices(total)` picks `num_frames` indices in `[0, total)`:
  uniform subsample if `total > num_frames`, edge-pad if `total < num_frames`.
- Transforms applied per frame independently. Augmentation in the train
  split (per-frame ColorJitter etc.) introduces sample-level rather than
  clip-level randomness; that's intentional and tolerable for this scale.

**`build_video_dataloaders(project_root, batch_size, num_frames, image_size, ...)`**
- Convenience to build all three video-level loaders.

### `src/models/resnet_classifier.py`

**`DeepfakeClassifier(pretrained=True, dropout=0.4, trainable_backbone_layers=1)`**
- ResNet-50 backbone (`IMAGENET1K_V2` weights when `pretrained=True`).
- Original `fc` is replaced by `Sequential(Dropout(dropout), Linear(2048, 1))`.
- `freeze_backbone(n)`: freezes all backbone params, unfreezes the trailing
  `n` of `[layer1, layer2, layer3, layer4]`. The fc head is always
  trainable.
- `forward(x)` returns `(B,)` logits (single-output binary).
- `train_step` / `val_step` are convenience wrappers used by `train_cnn.py`.

### `src/models/lstm_temporal.py`

**`LSTMTemporalClassifier(feature_dim=2048, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.5)`**
- `LSTM → temporal mean-pool → Dropout → Linear(H * dirs, 1)`.
- Mean-pool (rather than last hidden state) makes the head robust to
  variable, padded sequence lengths.
- Used as the temporal head inside `HybridCNNLSTM` and as the standalone
  head in `train_hybrid_v2.py`.

### `src/models/hybrid_model.py`

**`HybridCNNLSTM(pretrained=True, freeze_backbone=False, trainable_backbone_layers=4, lstm_hidden_size=256, lstm_num_layers=2, bidirectional=True, dropout=0.5)`**
- Composes a ResNet-50 backbone (output 2048-d per frame) with
  `LSTMTemporalClassifier`.
- `forward(clips)`: input shape `(B, T, 3, H, W)`, output `(B,)` logits.
- `_configure_backbone_grads`: same partial-unfreeze logic as
  `DeepfakeClassifier.freeze_backbone`, but on the hybrid's internal
  ResNet-50.

**`create_hybrid_model_and_optimizer(...)`**
- Factory: returns `(model, criterion=BCEWithLogitsLoss, optimizer=Adam)`.
- Used by `train_hybrid.py`.

### `src/preprocessing/`

- **`extract_frames(video_path, output_dir, n_frames=32, save=True)`** —
  uses OpenCV; seeks to `n_frames` evenly-spaced indices via
  `cv2.CAP_PROP_POS_FRAMES`. Returns BGR `np.ndarray` list. Handles videos
  shorter than `n_frames` by reading every available frame.
- **`FaceDetector(target_size=(224, 224), margin=20, device=None)`** —
  wraps `facenet_pytorch.MTCNN` for single-face detection. Returns BGR
  `np.ndarray` cropped to `target_size`, or `None` if no face is detected.
- **`create_splits.py`** — builds the train/val/test CSV splits with the
  60/15/15 ratio used in this project.
- **`run_pipeline.py`** — one-shot pipeline: raw video → extracted frames
  → detected/cropped faces → split CSV.

## Top-level scripts

### Training

| Script | Purpose | Output |
|---|---|---|
| `train_cnn.py` | Train `DeepfakeClassifier` end-to-end on frames | `models/cnn_baseline_best.pth` |
| `train_hybrid.py` | Train `HybridCNNLSTM` end-to-end on video clips (GPU recommended) | `models/hybrid_best.pth` |
| `train_hybrid_v2.py` | Train **only** the LSTM head on cached features (CPU-friendly) | `models/hybrid_head_only.pth` + `models/hybrid_best.pth` (assembled) |

`train_hybrid_v2.py` is the CPU path. It uses `WeightedRandomSampler` for
class balance, early-stops on val macro-F1 (not accuracy), and at the end
assembles the trained head with a fresh ImageNet ResNet-50 (or the trained
CNN backbone, depending on which feature cache was used) into a
`HybridCNNLSTM` checkpoint compatible with `evaluate_hybrid.py`.

### Feature caching

| Script | Purpose | Cache directory |
|---|---|---|
| `extract_features.py` | Cache **frozen ImageNet** ResNet-50 features per video | `data/features/` |
| `extract_features_cnn.py` | Cache features from the **trained CNN baseline** as backbone | `data/features_cnn/` |

Both produce one `(32, 2048)` `float32` `.npy` per video and a
`<split>_index.csv` per split. Re-runs skip already-cached files unless
`--overwrite` is passed.

### Evaluation

| Script | Purpose | Notes |
|---|---|---|
| `evaluate_cnn.py` | CNN baseline test eval at fixed threshold | Frame-level |
| `evaluate_hybrid.py` | Hybrid test eval | Runs full backbone + head on test clips |
| `evaluate_hybrid_cached.py` | Hybrid test eval reusing cached features | Fast (~30 s on CPU) |
| `tune_cnn_threshold.py` | Sweep CNN baseline threshold on val for macro-F1, re-evaluate test | Writes `outputs/cnn_evaluation_tuned.txt` |

### Demo launcher

- **`run_app.py`** — `python run_app.py` starts Flask on
  `127.0.0.1:5000` (or `APP_HOST` / `APP_PORT` env vars). Loads the
  predictor configured by `Config.MODEL_KIND`.

## Web app (`app/`)

### `app/config.py`

Central settings. Key fields:

| Field | Default | Notes |
|---|---|---|
| `MODEL_KIND` | `"cnn"` (env `APP_MODEL_KIND`) | `"cnn"` or `"hybrid"`. Decides which predictor is constructed at startup. |
| `CNN_MODEL_PATH` | `models/cnn_baseline_best.pth` | |
| `HYBRID_MODEL_PATH` | `models/hybrid_best.pth` | |
| `LSTM_HIDDEN_SIZE` / `LSTM_NUM_LAYERS` / `LSTM_BIDIRECTIONAL` / `LSTM_DROPOUT` | `64 / 1 / True / 0.6` | Architecture of the deployed hybrid head. Update if you retrain with a different shape. |
| `NUM_FRAMES` | `32` | Frames per video clip. |
| `IMAGE_SIZE` | `224` | Face crop size. |
| `MAX_CONTENT_LENGTH` | `50 * 1024 * 1024` | 50 MB upload cap. |
| `ALLOWED_EXTENSIONS` | `mp4, avi, mov, mkv, webm` | |
| `HOST` / `PORT` | `127.0.0.1` / `5000` (env `APP_HOST` / `APP_PORT`) | |
| `DEBUG` | `False` (env `APP_DEBUG=1` enables) | |

**`load_threshold(kind, default=None)`** — resolves the deployed decision
threshold by checking, in order: `outputs/threshold_<kind>.json`,
`outputs/threshold.json`, `Config.DEFAULT_THRESHOLD_BY_KIND[kind]`, then
`0.5`. The CNN tuner writes `outputs/threshold_cnn.json` so the deployed
threshold tracks the calibration result.

**`predictor_kwargs(kind)`** — returns the kwargs to construct the
selected predictor.

### `app/utils/preprocessor.py`

**`VideoPreprocessor(num_frames=32, image_size=224, face_margin=20, device=None)`**

`preprocess(video_path) -> (clip_tensor, info)`:
- Opens the video, validates it's readable and ≥0.5 s long.
- Calls `extract_frames` to get 32 evenly-spaced BGR frames.
- Runs MTCNN face detection on each. Pads with the last face if some
  frames had no detection.
- Builds `(1, T, 3, H, W)` normalised tensor + an `info` dict
  (`total_video_frames`, `video_fps`, `frames_with_faces`, `frames_padded`).

**`PreprocessingError(code, message)`** — raised on any user-facing
failure. The `code` is one of: `video_unreadable`, `video_empty`,
`video_too_short`, `frame_extraction_failed`, `no_faces`. The
`message` is the user-presentable string the API returns.

### `app/utils/predictor.py`

Two implementations behind a small factory:

**`CNNFramePredictor(checkpoint_path, dropout, trainable_backbone_layers, device)`**
- Loads `DeepfakeClassifier`, runs the full clip frame-by-frame
  (`logits(B*T,)` → `frame_probs(B, T)`).
- Video-level decision = `mean(frame_probs)` compared to threshold.
- Returns a result dict including per-frame probabilities so the UI can
  show which frames contributed to the decision.

**`HybridPredictor(checkpoint_path, lstm_hidden_size, lstm_num_layers, bidirectional, dropout, device)`**
- Loads `HybridCNNLSTM`, runs the whole clip in one forward pass.
- Returns a single video-level `prob_fake`.

**`build_predictor(kind, **kwargs)`** — returns `CNNFramePredictor` for
`"cnn"`, `HybridPredictor` for `"hybrid"`, raises `ValueError` otherwise.

Both predictors raise `ModelLoadError` on missing/invalid checkpoint, and
`PredictionError` on forward-pass exceptions or non-finite logits.

### `app/app.py`

Flask app factory. Sets up rotating-file loggers (`error.log` for `ERROR`-
level, `predictions.log` for prediction outcomes), then defines:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Upload page |
| `/about` | GET | Model info page |
| `/predict` | POST | Run inference on uploaded video; returns JSON |
| `/health` | GET | Liveness probe (`{"ok": True}` if model loaded) |

`/predict` JSON shape on success:

```json
{
  "ok": true,
  "request_id": "85337c39",
  "label": "FAKE",
  "confidence_pct": 99.13,
  "probability_fake_pct": 99.13,
  "probability_real_pct": 0.87,
  "threshold": 0.75,
  "device": "cpu",
  "inference_seconds": 3.93,
  "total_seconds": 13.56,
  "video_info": {
    "total_video_frames": 714,
    "video_fps": 25.0,
    "frames_sampled": 32,
    "frames_with_faces": 32,
    "frames_padded": 0
  }
}
```

On failure:

```json
{ "ok": false, "error": { "code": "no_faces", "message": "..." } }
```

Error codes returned: `missing_file`, `empty_filename`, `invalid_type`,
`empty_file`, `file_too_large`, `save_failed`, `video_unreadable`,
`video_empty`, `video_too_short`, `frame_extraction_failed`, `no_faces`,
`inference_failed`, `model_not_loaded`, `internal_error`, `not_found`.

### Frontend (`app/templates/`, `app/static/`)

- `templates/index.html` — drag-and-drop or click-to-pick upload, status
  spinner, result card with confidence bar and metadata table.
- `templates/about.html` — architecture, performance numbers, training
  pipeline, limitations. Branched on `MODEL_KIND` so it shows the
  deployed model's specifics.
- `templates/error.html` — fallback page for direct GETs to error states.
- `static/css/style.css` — navy `#1E2761` + teal `#028090` palette,
  responsive (mobile-first via `@media (max-width: 600px)`).
- `static/js/main.js` — drag/drop wiring, AJAX submit, in-place result
  rendering, no jQuery / no build step.

## Configuration knobs at a glance

| Variable | Default | What it does |
|---|---|---|
| `APP_MODEL_KIND` | `cnn` | switch between deployed CNN / hybrid |
| `APP_HOST` | `127.0.0.1` | bind address |
| `APP_PORT` | `5000` | bind port |
| `APP_DEBUG` | `0` | `1` enables Flask debug mode |
| `APP_SECRET_KEY` | dev placeholder | override before any production use |

## Logging

- `app/logs/error.log` — `ERROR`-level only; `RotatingFileHandler`
  (1 MB × 3 backups).
- `app/logs/predictions.log` — `INFO`-level; one line per prediction
  request (success or `PREPROCESS_ERROR` / `PREDICTION_ERROR`).
  `RotatingFileHandler` (2 MB × 5 backups).
- Console mirror of both streams.

Each request gets an 8-char `request_id` that appears in both the
log line and the JSON response — useful for grepping when a user reports
a problem.
