"""
Train the end-to-end CNN-LSTM hybrid (Week 4).

Default hyperparameters target a Colab T4. For a quick local sanity run use
e.g. `--epochs 1 --batch-size 2 --num-frames 8 --num-workers 0`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.video_dataset import build_video_dataloaders
from src.models.hybrid_model import create_hybrid_model_and_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 4 hybrid CNN-LSTM training.")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze the CNN entirely (feature-extractor mode).")
    parser.add_argument("--trainable-backbone-layers", type=int, default=4,
                        help="0..4: number of trailing ResNet stages to keep trainable "
                             "when --freeze-backbone is not set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--disable-pretrained", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--checkpoint-name", type=str, default="hybrid_best.pth")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        m = model.train_step(batch, criterion, optimizer, device)
        running_loss += m["loss"]
        running_acc += m["accuracy"]
        n += 1
    return running_loss / max(n, 1), running_acc / max(n, 1)


def validate_one_epoch(model, loader, criterion, device) -> tuple[float, float]:
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        m = model.val_step(batch, criterion, device)
        running_loss += m["loss"]
        running_acc += m["accuracy"]
        n += 1
    return running_loss / max(n, 1), running_acc / max(n, 1)


def save_checkpoint(path: Path, model, optimizer, epoch, best_val_acc, history, args) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
            "config": vars(args),
        },
        path,
    )


def plot_curves(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Hybrid Loss"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Hybrid Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(args.project_root).resolve()
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = build_video_dataloaders(
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        image_size=args.image_size,
    )
    print(
        f"Train videos: {len(train_loader.dataset)} | "
        f"Val videos: {len(val_loader.dataset)} | "
        f"Frames/clip: {args.num_frames}"
    )

    model, criterion, optimizer = create_hybrid_model_and_optimizer(
        device=device,
        pretrained=not args.disable_pretrained,
        freeze_backbone=args.freeze_backbone,
        trainable_backbone_layers=args.trainable_backbone_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        bidirectional=not args.no_bidirectional,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    start_epoch = 1
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    if args.resume:
        ck = torch.load(Path(args.resume), map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = int(ck["epoch"]) + 1
        best_val_acc = float(ck.get("best_val_acc", 0.0))
        history = ck.get("history", history)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    checkpoint_path = models_dir / args.checkpoint_name
    history_path = outputs_dir / "train_history_hybrid.json"
    curve_path = outputs_dir / "hybrid_training_curves.png"

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_acc, history, args)
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"Early stopping: no val_acc improvement for "
                    f"{args.early_stop_patience} epochs."
                )
                break

    plot_curves(history, curve_path)
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best model: {checkpoint_path}")
    print(f"Training curves: {curve_path}")
    print(f"History JSON: {history_path}")


if __name__ == "__main__":
    main()
