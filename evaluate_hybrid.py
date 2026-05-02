"""
Evaluate the trained CNN-LSTM hybrid on the test split (video-level metrics).
"""

from __future__ import annotations

import argparse
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

from src.data.video_dataset import VideoSequenceDataset
from src.models.hybrid_model import HybridCNNLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid CNN-LSTM on test split.")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)
    return parser.parse_args()


def load_hybrid(checkpoint_path: Path, device: torch.device, args) -> HybridCNNLSTM:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HybridCNNLSTM(
        pretrained=False,  # weights replaced by checkpoint
        freeze_backbone=False,
        trainable_backbone_layers=4,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        bidirectional=not args.no_bidirectional,
        dropout=args.dropout,
    ).to(device)
    state_dict = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(model: HybridCNNLSTM, loader: DataLoader, device: torch.device, threshold: float) -> dict:
    targets, preds, probs = [], [], []
    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="Evaluating", leave=False):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)
            logits = model(clips)
            p = torch.sigmoid(logits)
            targets.extend(labels.long().cpu().tolist())
            probs.extend(p.cpu().tolist())
            preds.extend((p >= threshold).long().cpu().tolist())

    y_true = np.array(targets)
    y_pred = np.array(preds)
    y_prob = np.array(probs)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["real", "fake"], digits=4, zero_division=0
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Hybrid CNN-LSTM - Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Pred Real", "Pred Fake"])
    plt.yticks(ticks, ["True Real", "True Fake"])
    thr = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thr else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_text_report(metrics: dict, report_path: Path, checkpoint_path: Path, threshold: float) -> None:
    cm = metrics["confusion_matrix"]
    lines = [
        "HYBRID CNN-LSTM EVALUATION REPORT",
        "=" * 40,
        f"Checkpoint: {checkpoint_path}",
        f"Threshold : {threshold:.2f}",
        "",
        f"Accuracy : {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall   : {metrics['recall']:.4f}",
        f"F1-Score : {metrics['f1_score']:.4f}",
        f"ROC-AUC  : {metrics['roc_auc']:.4f}",
        "",
        "Confusion Matrix [[TN, FP], [FN, TP]]:",
        str(cm.tolist()),
        "",
        "Classification Report:",
        str(metrics["classification_report"]),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else (project_root / "models" / "hybrid_best.pth").resolve()
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = VideoSequenceDataset(
        csv_path=project_root / "data" / "splits" / "test.csv",
        project_root=project_root,
        split="test",
        num_frames=args.num_frames,
        image_size=args.image_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_hybrid(checkpoint_path, device, args)
    metrics = evaluate(model, test_loader, device, args.threshold)

    report_path = outputs_dir / "hybrid_evaluation.txt"
    cm_path = outputs_dir / "hybrid_confusion_matrix.png"
    save_text_report(metrics, report_path, checkpoint_path, args.threshold)
    save_confusion_matrix(metrics["confusion_matrix"], cm_path)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix plot: {cm_path}")


if __name__ == "__main__":
    main()
