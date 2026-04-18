from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.dataset import build_dataloaders
from src.models.resnet_classifier import create_model_and_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 3 CNN baseline (local quick training).")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--trainable-backbone-layers", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=8)
    parser.add_argument("--train-max-samples", type=int, default=4096)
    parser.add_argument("--val-max-samples", type=int, default=1024)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--disable-pretrained", action="store_true")
    parser.add_argument("--full-data", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
) -> tuple[float, float]:
    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    if grad_accum_steps <= 1:
        for batch in tqdm(loader, desc="Train", leave=False):
            metrics = model.train_step(batch, criterion, optimizer, device)
            running_loss += metrics["loss"]
            running_acc += metrics["accuracy"]
            total_batches += 1
        return running_loss / max(total_batches, 1), running_acc / max(total_batches, 1)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        logits = model(images)
        loss = criterion(logits, labels)
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean().item()

        running_loss += loss.item()
        running_acc += acc
        total_batches += 1

    return running_loss / max(total_batches, 1), running_acc / max(total_batches, 1)


def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        metrics = model.val_step(batch, criterion, device)
        running_loss += metrics["loss"]
        running_acc += metrics["accuracy"]
        total_batches += 1

    return running_loss / max(total_batches, 1), running_acc / max(total_batches, 1)


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    history: dict[str, list[float]],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "config": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)


def plot_training_curves(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

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

    if args.full_data:
        args.max_frames_per_video = None
        args.train_max_samples = None
        args.val_max_samples = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = build_dataloaders(
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        max_frames_per_video=args.max_frames_per_video,
        max_train_samples=args.train_max_samples,
        max_val_samples=args.val_max_samples,
        max_test_samples=256 if not args.full_data else None,
    )
    print(
        f"Train samples: {len(train_loader.dataset)} | "
        f"Val samples: {len(val_loader.dataset)}"
    )

    model, criterion, optimizer = create_model_and_optimizer(
        device=device,
        pretrained=not args.disable_pretrained,
        dropout=args.dropout,
        trainable_backbone_layers=args.trainable_backbone_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val_acc = 0.0
    start_epoch = 1
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
        history = checkpoint.get("history", history)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    checkpoint_path = models_dir / "cnn_baseline_best.pth"
    history_path = outputs_dir / "train_history_local.json"
    curve_path = outputs_dir / "training_curves.png"

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
        )
        val_loss, val_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                history=history,
                args=args,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

    plot_training_curves(history=history, out_path=curve_path)
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best model: {checkpoint_path}")
    print(f"Training curves: {curve_path}")
    print(f"History JSON: {history_path}")


if __name__ == "__main__":
    main()

