"""
Fast hybrid-LSTM evaluation using cached features.

Loads the trained LSTM head only (`models/hybrid_head_only.pth`) and runs it
on cached features for the test split. Computes metrics at threshold 0.5 and
also picks the val-macro-F1-optimal threshold and re-evaluates test there.
Avoids running ResNet-50 on test frames again (which would take ~15 min on CPU).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_temporal import LSTMTemporalClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid (LSTM head) on cached features.")
    parser.add_argument("--head-checkpoint", type=str,
                        default=str(PROJECT_ROOT / "models" / "hybrid_head_only.pth"))
    parser.add_argument("--features-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "features"))
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=37)
    return parser.parse_args()


def load_features(features_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    index_path = features_dir / f"{split}_index.csv"
    df = pd.read_csv(index_path)
    feats: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    for _, row in df.iterrows():
        path = features_dir / split / f"{row['video_id']}.npy"
        feats.append(np.load(path))
        labels.append(int(row["label"]))
        ids.append(str(row["video_id"]))
    X = np.stack(feats, axis=0)  # (N, T, 2048)
    y = np.array(labels)
    return X, y, ids


def head_from_checkpoint(ck_path: Path) -> tuple[LSTMTemporalClassifier, dict]:
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    head = LSTMTemporalClassifier(
        feature_dim=2048,
        hidden_size=int(cfg.get("lstm_hidden_size", 256)),
        num_layers=int(cfg.get("lstm_num_layers", 2)),
        bidirectional=not bool(cfg.get("no_bidirectional", False)),
        dropout=float(cfg.get("dropout", 0.5)),
    )
    state_dict = ck["head_state_dict"] if "head_state_dict" in ck else ck.get("model_state_dict", ck)
    head.load_state_dict(state_dict)
    head.eval()
    return head, cfg


def collect_probs(head: LSTMTemporalClassifier, X: np.ndarray) -> np.ndarray:
    probs: list[float] = []
    bs = 32
    with torch.no_grad():
        for start in range(0, len(X), bs):
            batch = torch.from_numpy(X[start : start + bs])
            logits = head(batch)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
    return np.array(probs)


def metrics_at(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_real": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_real": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_real": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def write_report(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
    val_macro_f1: float, head_path: Path, report_path: Path, cm_path: Path,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    lines = [
        "HYBRID CNN-LSTM EVALUATION REPORT (cached-features path)",
        "=" * 56,
        f"Head checkpoint: {head_path}",
        f"Threshold      : {threshold:.3f}",
        f"Val macro-F1   : {val_macro_f1:.4f}",
        f"Test ROC-AUC   : {roc_auc:.4f}",
        "",
        f"Test accuracy  : {accuracy_score(y_true, y_pred):.4f}",
        f"Macro F1       : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        f"F1 real        : {f1_score(y_true, y_pred, pos_label=0, zero_division=0):.4f}",
        f"F1 fake        : {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}",
        "",
        "Confusion Matrix [[TN, FP], [FN, TP]]:",
        str(cm.tolist()),
        "",
        classification_report(y_true, y_pred, target_names=["real", "fake"], digits=4, zero_division=0),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Hybrid (thr={threshold:.2f}) — Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Pred Real", "Pred Fake"])
    plt.yticks(ticks, ["True Real", "True Fake"])
    color_thr = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > color_thr else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    features_dir = Path(args.features_dir)
    head_path = Path(args.head_checkpoint)
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    head, cfg = head_from_checkpoint(head_path)
    print(f"Loaded head from {head_path}")
    print(f"Head config: hidden={cfg.get('lstm_hidden_size')} "
          f"layers={cfg.get('lstm_num_layers')} "
          f"bidir={not cfg.get('no_bidirectional', False)} "
          f"dropout={cfg.get('dropout')}")

    print("Loading val features...")
    Xv, yv, _ = load_features(features_dir, "val")
    print("Loading test features...")
    Xt, yt, _ = load_features(features_dir, "test")
    print(f"Val: {len(yv)} videos | Test: {len(yt)} videos")

    pv = collect_probs(head, Xv)
    pt = collect_probs(head, Xt)

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    sweep_rows = [metrics_at(yv, pv, t) for t in thresholds]
    best_row = max(sweep_rows, key=lambda r: r["macro_f1"])
    best_thr = best_row["threshold"]

    sweep_csv = outputs_dir / "hybrid_threshold_sweep.csv"
    with sweep_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader(); writer.writerows(sweep_rows)

    plt.figure(figsize=(7, 4))
    plt.plot([r["threshold"] for r in sweep_rows], [r["macro_f1"] for r in sweep_rows], label="macro-F1")
    plt.plot([r["threshold"] for r in sweep_rows], [r["f1_real"] for r in sweep_rows], label="F1 real")
    plt.plot([r["threshold"] for r in sweep_rows], [r["f1_fake"] for r in sweep_rows], label="F1 fake")
    plt.axvline(best_thr, color="red", linestyle="--", label=f"best={best_thr:.2f}")
    plt.xlabel("Threshold"); plt.ylabel("Score (val)")
    plt.title("Hybrid — Threshold Sweep (val)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(outputs_dir / "hybrid_threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    test_at_default = metrics_at(yt, pt, 0.5)
    test_at_tuned = metrics_at(yt, pt, best_thr)

    print("\n=== Hybrid (cached-features) results ===")
    print(f"Best val threshold: {best_thr:.3f} (val macro-F1={best_row['macro_f1']:.4f})")
    for label, m in [("Test @ 0.50", test_at_default), (f"Test @ {best_thr:.3f}", test_at_tuned)]:
        print(f"\n{label}:")
        for k in ("accuracy", "macro_f1", "f1_real", "f1_fake", "recall_real", "recall_fake"):
            print(f"  {k:14s} = {m[k]:.4f}")
    try:
        roc_auc = float(roc_auc_score(yt, pt))
    except ValueError:
        roc_auc = float("nan")
    print(f"\nTest ROC-AUC: {roc_auc:.4f}")

    write_report(
        y_true=yt, y_prob=pt, threshold=best_thr,
        val_macro_f1=best_row["macro_f1"],
        head_path=head_path,
        report_path=outputs_dir / "hybrid_evaluation.txt",
        cm_path=outputs_dir / "hybrid_confusion_matrix.png",
    )
    write_report(
        y_true=yt, y_prob=pt, threshold=0.5,
        val_macro_f1=best_row["macro_f1"],
        head_path=head_path,
        report_path=outputs_dir / "hybrid_evaluation_default.txt",
        cm_path=outputs_dir / "hybrid_confusion_matrix_default.png",
    )

    threshold_path = outputs_dir / "threshold.json"
    threshold_path.write_text(json.dumps({
        "threshold": best_thr,
        "val_macro_f1": best_row["macro_f1"],
        "test_macro_f1_at_tuned": test_at_tuned["macro_f1"],
    }, indent=2), encoding="utf-8")

    print(f"\nWrote:")
    print(f"  {outputs_dir / 'hybrid_evaluation.txt'}")
    print(f"  {outputs_dir / 'hybrid_evaluation_default.txt'}")
    print(f"  {outputs_dir / 'hybrid_threshold_sweep.csv'}")
    print(f"  {outputs_dir / 'hybrid_threshold_sweep.png'}")
    print(f"  {threshold_path}")


if __name__ == "__main__":
    main()
