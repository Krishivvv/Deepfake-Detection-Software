# Veridex — Prioritized Improvement Plan

Status legend: ✅ done in this pass · ⬜ proposed. Estimates assume one engineer.

## P1 — Do now (correctness, hygiene, shippability)

| # | Item | Files | Est. | Status |
|---|---|---|---|---|
| 1 | Remove hard-coded machine paths; central `config.yaml` + loader | `config.yaml`, `src/config.py`, `notebooks/Train_Hybrid.ipynb` | 2h | ✅ |
| 2 | Consolidate 3 eval scripts → `evaluate.py --model {cnn,hybrid,hybrid_v3}` | `evaluate.py`, `scripts/archive/` | 3h | ✅ |
| 3 | Consolidate 2 feature extractors → `extract_features.py --backbone {imagenet,cnn}` | `extract_features.py` | 1h | ✅ |
| 4 | Shared metrics logger (acc/P/R/F1 + CM → `outputs/`) | `src/evaluation/metrics.py` | 2h | ✅ |
| 5 | Stop tracking `app/users.db` | git index | 5m | ✅ |
| 6 | Tests: loader shapes/labels, preprocessing parity, forward-pass smoke | `tests/` | 3h | ✅ |
| 7 | CI (ruff + pytest, py3.10/3.11) | `.github/workflows/ci.yml` | 1h | ✅ |
| 8 | MODEL_CARD.md + README metrics/demo section | `MODEL_CARD.md`, `README.md` | 2h | ✅ |
| 9 | Gradio inference demo (+ Grad-CAM) & Space config | `gradio_app.py`, `requirements-space.txt`, `README_HF_SPACE.md`, `Dockerfile.space` | 4h | ✅ (deploy pending CHECKPOINT B) |
| 10 | Host weights on HF Hub; set live-demo link in README | HF Hub, `README.md` | 1h | ⬜ (CHECKPOINT B) |
| 11 | Set GitHub repo description + topics | GitHub settings / `gh` | 10m | ⬜ (commands in `DEPLOYMENT.md §6`) |

## P2 — Next (robustness, reproducibility, quality)

| # | Item | Files | Est. | Status |
|---|---|---|---|---|
| 1 | Fully lock deps with `pip-compile` (`requirements.lock`) | `requirements*.txt` | 1h | ⬜ |
| 2 | Add `--seed`/deterministic flags to `evaluate.py`; log git SHA in reports | `evaluate.py`, `src/evaluation/metrics.py` | 1h | ⬜ |
| 3 | Replace deprecated `torch.load(weights_only=False)` with safe loaders / `safetensors` | `app/utils/predictor.py`, `evaluate.py` | 2h | ⬜ |
| 4 | Integration test that loads real checkpoints (gated by `models/` presence) | `tests/test_predictor_integration.py` | 2h | ⬜ |
| 5 | Frontend CI (`npm run build` + lint) job | `.github/workflows/ci.yml` | 1h | ⬜ |
| 6 | Pre-commit (ruff + end-of-file/whitespace) | `.pre-commit-config.yaml` | 30m | ⬜ |
| 7 | Calibration: report Brier/ECE alongside threshold; per-manipulation breakdown | `src/evaluation/metrics.py`, `evaluate.py` | 3h | ⬜ |

## P3 — Later (capability, polish)

| # | Item | Files | Est. | Status |
|---|---|---|---|---|
| 1 | Image-only mode in the demo (single-frame fallback when no video) | `gradio_app.py` | 2h | ⬜ |
| 2 | End-to-end GPU fine-tune to chase 90 %+; document in MODEL_CARD | training scripts | 1–2d | ⬜ |
| 3 | Test-time augmentation / temporal smoothing for stability | `app/utils/predictor.py` | 4h | ⬜ |
| 4 | ONNX export + quantization for faster CPU inference | new `export.py` | 4h | ⬜ |
| 5 | Type-check gate (`mypy`/`pyright`) in CI | CI, annotations | 3h | ⬜ |
| 6 | Architecture diagram + short demo GIF in README | `docs/` | 2h | ⬜ |

## Quick reference — canonical pipeline (post-refactor)
```bash
python extract_features.py --backbone cnn      # cache backbone features
python train_hybrid_v2.py                       # train LSTM head -> hybrid_v3_head.pth
python evaluate.py --model hybrid_v3            # metrics + threshold -> outputs/
python gradio_app.py                            # local demo (Grad-CAM)
python cli.py serve                             # full Flask + Next.js backend
```
