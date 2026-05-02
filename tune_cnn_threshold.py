"""
Tune the CNN baseline's decision threshold without retraining.

The existing cnn_baseline_best.pth predicts almost everything as 'fake' at the
default threshold of 0.5 (recall on real = 5.5%). The model itself may still
have signal — the head is just poorly calibrated under the 4:1 fake:real
class imbalance. We sweep thresholds on the validation split, pick the one
that maximises macro-F1 (so both classes get equal weight), then re-evaluate
on the test split with both 0.5 and the tuned threshold.

Outputs:
    outputs/cnn_threshold_sweep.csv      # threshold, val_macro_f1, ...
    outputs/cnn_evaluation_tuned.txt     # test report at tuned threshold
    outputs/cnn_confusion_matrix_tuned.png
    outputs/cnn_threshold_sweep.png      # macro-F1 vs threshold curve
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DeepfakeDataset  # noqa: E402
from src.models.resnet_classifier import DeepfakeClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold-tune CNN baseline (no retrain).")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--trainable-backbone-layers", type=int, default=1)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=37)
    return parser.parse_args()


def collect_probs(model: DeepfakeClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    targets: list[int] = []
    probs: list[float] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="infer", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)
            logits = model(images)
            p = torch.sigmoid(logits)
            targets.extend(labels.long().cpu().tolist())
            probs.extend(p.cpu().tolist())
    return np.array(targets), np.array(probs)


def metrics_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_real": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_real": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_real": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def write_report(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
    checkpoint_path: Path, report_path: Path, cm_image_path: Path,
    val_macro_f1: float, roc_auc: float,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["real", "fake"], digits=4, zero_division=0)
    lines = [
        "CNN BASELINE EVALUATION REPORT (THRESHOLD-TUNED)",
        "=" * 50,
        f"Checkpoint     : {checkpoint_path}",
        f"Threshold      : {threshold:.3f}  (selected to maximise val macro-F1)",
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
        "Classification Report:",
        str(report),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"CNN (tuned thr={threshold:.2f}) - Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Pred Real", "Pred Fake"])
    plt.yticks(ticks, ["True Real", "True Fake"])
    thr_color = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thr_color else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_image_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else (project_root / "models" / "cnn_baseline_best.pth").resolve()
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = DeepfakeDataset(
        csv_path=project_root / "data" / "splits" / "val.csv",
        project_root=project_root,
        split="val",
        image_size=args.image_size,
    )
    test_dataset = DeepfakeDataset(
        csv_path=project_root / "data" / "splits" / "test.csv",
        project_root=project_root,
        split="test",
        image_size=args.image_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    ck = torch.load(checkpoint_path, map_location=device)
    model = DeepfakeClassifier(
        pretrained=False,
        dropout=args.dropout,
        trainable_backbone_layers=args.trainable_backbone_layers,
    ).to(device)
    state_dict = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    print("Collecting val probabilities...")
    y_val, p_val = collect_probs(model, val_loader, device)
    print("Collecting test probabilities...")
    y_test, p_test = collect_probs(model, test_loader, device)

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    sweep_rows = [metrics_at(y_val, p_val, t) for t in thresholds]
    best_row = max(sweep_rows, key=lambda r: r["macro_f1"])
    best_thr = best_row["threshold"]

    sweep_csv = outputs_dir / "cnn_threshold_sweep.csv"
    with sweep_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    plt.figure(figsize=(7, 4))
    plt.plot([r["threshold"] for r in sweep_rows], [r["macro_f1"] for r in sweep_rows], label="macro-F1")
    plt.plot([r["threshold"] for r in sweep_rows], [r["f1_real"] for r in sweep_rows], label="F1 real")
    plt.plot([r["threshold"] for r in sweep_rows], [r["f1_fake"] for r in sweep_rows], label="F1 fake")
    plt.axvline(best_thr, color="red", linestyle="--", label=f"best={best_thr:.2f}")
    plt.xlabel("Threshold"); plt.ylabel("Score (val)")
    plt.title("CNN Baseline – Threshold Sweep (val)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(outputs_dir / "cnn_threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    try:
        roc_auc = float(roc_auc_score(y_test, p_test))
    except ValueError:
        roc_auc = float("nan")

    report_path = outputs_dir / "cnn_evaluation_tuned.txt"
    cm_image_path = outputs_dir / "cnn_confusion_matrix_tuned.png"
    write_report(
        y_true=y_test, y_prob=p_test, threshold=best_thr,
        checkpoint_path=checkpoint_path,
        report_path=report_path, cm_image_path=cm_image_path,
        val_macro_f1=best_row["macro_f1"], roc_auc=roc_auc,
    )

    print(f"\nBest threshold (val macro-F1={best_row['macro_f1']:.4f}): {best_thr:.3f}")
    print(f"Test @ tuned threshold:")
    test_metrics = metrics_at(y_test, p_test, best_thr)
    for k in ("accuracy", "macro_f1", "f1_real", "f1_fake", "precision_real", "recall_real"):
        print(f"  {k:14s} = {test_metrics[k]:.4f}")
    print(f"Test ROC-AUC      = {roc_auc:.4f}")
    print(f"\nReports:")
    print(f"  {report_path}")
    print(f"  {cm_image_path}")
    print(f"  {sweep_csv}")
    print(f"  {outputs_dir / 'cnn_threshold_sweep.png'}")


if __name__ == "__main__":
    main()
