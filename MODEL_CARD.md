# Model Card — Veridex `hybrid_v3`

## Model summary
Veridex `hybrid_v3` is a **binary deepfake video classifier** (real = 0,
fake = 1). It is a two-stage model:

1. **Spatial backbone** — ResNet-50 (ImageNet-initialised, last block
   fine-tuned on FaceForensics++) used as a frozen 2048-d per-frame feature
   extractor (`fc → Identity`). This is the trained `cnn_baseline_best.pth`
   backbone, *not* raw ImageNet weights.
2. **Temporal head** — a single-layer **BiLSTM** (hidden 128, dropout 0.5) with
   temporal mean-pooling and a linear sigmoid output, trained on the cached
   backbone features (`hybrid_v3_head.pth`).

A video is decoded → 32 frames sampled → faces cropped (MTCNN) → ImageNet
normalized → fed as a sequence → single video-level fake probability.

- **Task:** frame-sequence (video-level) deepfake detection
- **Input:** video clip (≥1 s, visible face)
- **Output:** `REAL`/`FAKE` + calibrated probability
- **Decision threshold:** **0.575** (selected to maximise val macro-F1)
- **Framework:** PyTorch 2.x · **License:** MIT

## Intended use
- **In scope:** education, portfolio demonstration, research baselines,
  triage/"second opinion" on face-centric clips similar to FaceForensics++.
- **Out of scope:** forensic or legal evidence; authentication decisions;
  non-face / multi-face / audio deepfakes; any high-stakes automated action.

## Training data
- **Dataset:** [FaceForensics++](https://github.com/ondyari/FaceForensics)
  (Rössler et al., ICCV 2019), a ~700-video subset spanning real videos and
  the Deepfakes / Face2Face / FaceSwap / NeuralTextures manipulations.
- **Splits:** video-level train/val/test in `data/splits/*.csv` (no video
  appears in two splits — split is by source video to avoid leakage).
- **Class balance:** ~4:1 fake:real; corrected during head training with a
  `WeightedRandomSampler` and macro-F1 early stopping.
- The dataset itself is **not** redistributed in this repo (license + size).

## Preprocessing (training = inference)
| Step | Value |
|---|---|
| Frames per video | 32 (uniformly sampled) |
| Face detection/crop | MTCNN (`facenet-pytorch`), margin 20 px |
| Resize | 224 × 224 |
| Normalize | ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` |
| Train-only augmentation | h-flip, ±10° rotation, colour jitter |

Inference reuses the exact eval transform (no augmentation), guaranteeing no
train/serve skew. See `src/data/dataset.py` and `app/utils/preprocessor.py`.

## Evaluation results
Held-out **test set: 150 videos** (120 fake / 30 real). Reproduce with
`python evaluate.py --model hybrid_v3` (verified to reproduce these numbers
from cached features).

| Metric | @ thr 0.575 (deployed) | @ thr 0.50 |
|---|---|---|
| Accuracy | **0.800** | 0.820 |
| ROC-AUC | **0.870** | 0.870 |
| Macro-F1 | 0.731 | 0.751 |
| F1 (fake) | 0.867 | 0.882 |
| F1 (real) | 0.595 | 0.620 |
| Recall (real) | 0.733 | 0.733 |
| Recall (fake) | 0.817 | 0.842 |

Confusion matrix @ 0.575 `[[TN, FP],[FN, TP]]`: `[[22, 8], [22, 98]]`.

**Baseline comparison** — frame-level CNN only (`evaluate.py --model cnn`):
test acc 0.776, ROC-AUC 0.766 @ thr 0.75. The temporal head adds ~+10 ROC-AUC
points.

## Limitations & biases
- **Real-class recall ≈ 73 %** — authentic videos are sometimes flagged fake;
  do not use as sole arbiter.
- Trained on a 700-video subset; published SOTA (90 %+) fine-tunes end-to-end
  on GPU with far more data.
- Single-modality, single-face, no audio / compression-domain cues.
- Generalisation to manipulation methods **outside** FaceForensics++
  (e.g. modern diffusion face-swaps) is **unverified** and likely weaker.
- Inherits any demographic/source biases in FaceForensics++.

## Ethical considerations
Deepfake detectors can produce false accusations. Veridex outputs a
probability, not a determination, and is **not for forensic use**. Treat
`FAKE` as "warrants human review," never as proof.

## Files
| File | Role | Size |
|---|---|---|
| `cnn_baseline_best.pth` | ResNet-50 backbone | ~214 MB |
| `hybrid_v3_head.pth` | BiLSTM temporal head | ~9 MB |

Weights are hosted on the Hugging Face Hub (see [DEPLOYMENT.md](DEPLOYMENT.md)),
never committed to git.

## Citation
```
@inproceedings{roessler2019faceforensicspp,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={R\"ossler, Andreas and others},
  booktitle={ICCV}, year={2019}
}
```
