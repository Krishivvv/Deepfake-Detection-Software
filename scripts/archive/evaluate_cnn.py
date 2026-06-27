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
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import DeepfakeDataset
from src.models.resnet_classifier import DeepfakeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN baseline on test split.")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--trainable-backbone-layers", type=int, default=1)
    parser.add_argument("--disable-pretrained", action="store_true")
    return parser.parse_args()


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    pretrained: bool,
    dropout: float,
    trainable_backbone_layers: int,
) -> DeepfakeClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DeepfakeClassifier(
        pretrained=pretrained,
        dropout=dropout,
        trainable_backbone_layers=trainable_backbone_layers,
    ).to(device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(
    model: DeepfakeClassifier,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, object]:
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            all_targets.extend(labels.long().cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

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
    return metrics


def save_confusion_matrix_image(conf_mat: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("CNN Baseline - Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred Real", "Pred Fake"])
    plt.yticks(tick_marks, ["True Real", "True Fake"])

    threshold = conf_mat.max() / 2.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(
                j,
                i,
                format(conf_mat[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_mat[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_text_report(
    metrics: dict[str, object],
    report_path: Path,
    checkpoint_path: Path,
    threshold: float,
) -> None:
    conf_mat = metrics["confusion_matrix"]
    lines = [
        "CNN BASELINE EVALUATION REPORT",
        "=" * 40,
        f"Checkpoint: {checkpoint_path}",
        f"Threshold : {threshold:.2f}",
        "",
        f"Accuracy : {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall   : {metrics['recall']:.4f}",
        f"F1-Score : {metrics['f1_score']:.4f}",
        "",
        "Confusion Matrix [[TN, FP], [FN, TP]]:",
        str(conf_mat.tolist()),
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
        else (project_root / "models" / "cnn_baseline_best.pth").resolve()
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = DeepfakeDataset(
        csv_path=project_root / "data" / "splits" / "test.csv",
        project_root=project_root,
        split="test",
        image_size=args.image_size,
        max_frames_per_video=None,
        max_samples=None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        device=device,
        pretrained=not args.disable_pretrained,
        dropout=args.dropout,
        trainable_backbone_layers=args.trainable_backbone_layers,
    )
    metrics = evaluate(model=model, test_loader=test_loader, device=device, threshold=args.threshold)

    report_path = outputs_dir / "cnn_evaluation.txt"
    cm_path = outputs_dir / "cnn_confusion_matrix.png"
    save_text_report(metrics=metrics, report_path=report_path, checkpoint_path=checkpoint_path, threshold=args.threshold)
    save_confusion_matrix_image(metrics["confusion_matrix"], cm_path)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix plot: {cm_path}")


if __name__ == "__main__":
    main()

