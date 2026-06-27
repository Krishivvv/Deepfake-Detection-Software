# Archived scripts

These scripts were superseded during the consolidation refactor. They are kept
for provenance and reproducibility of past results, but are **not** part of the
maintained pipeline. Prefer the consolidated entry points at the repo root.

| Archived script | Replaced by | Notes |
|---|---|---|
| `evaluate_cnn.py` | `evaluate.py --model cnn` | Frame-level CNN test eval. |
| `evaluate_hybrid.py` | `evaluate.py --model hybrid` | End-to-end CNN-LSTM test eval. |
| `evaluate_hybrid_cached.py` | `evaluate.py --model hybrid_v3` | Deployed model; cached-feature eval + val threshold sweep. |
| `extract_features_cnn.py` | `extract_features.py --backbone cnn` | Trained CNN-baseline backbone features. |
| `train_hybrid.py` | `train_hybrid_v2.py` | Original end-to-end hybrid trainer; v2 (cached-feature head) produces the deployed `hybrid_v3` head. |

The consolidated `evaluate.py` reproduces the published `hybrid_v3` numbers
exactly (acc 0.800, macro-F1 0.731, ROC-AUC 0.870 @ threshold 0.575).

The canonical training pipeline is:

1. `python extract_features.py --backbone cnn`  → cached features
2. `python train_hybrid_v2.py`                  → `hybrid_v3_head.pth`
3. `python evaluate.py --model hybrid_v3`       → metrics + threshold
