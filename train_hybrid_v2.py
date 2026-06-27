"""
Train the LSTM temporal head on cached ResNet-50 features (Phase 2).

Run `extract_features.py --backbone cnn` first. This script trains only
LSTMTemporalClassifier on the cached features (CPU-friendly, ~10-20 min for
20 epochs on 700 videos), then assembles the trained head with a fresh
ImageNet-pretrained ResNet-50 backbone into a HybridCNNLSTM checkpoint, and
saves the head-only weights consumed by `evaluate.py --model hybrid_v3`.

Class imbalance (4:1 fake:real) is corrected via WeightedRandomSampler.
Early stopping uses macro-F1 on val so a "predict-everything-fake" collapse
is penalised instead of rewarded.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid_model import HybridCNNLSTM  # noqa: E402
from src.models.lstm_temporal import LSTMTemporalClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM head on cached features.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--features-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--checkpoint-name", type=str, default="hybrid_best.pth")
    parser.add_argument("--head-only-checkpoint", type=str, default="hybrid_head_only.pth",
                        help="Smaller checkpoint with just the trained LSTM head; "
                             "useful for rapid re-evaluation.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CachedFeatureDataset(Dataset):
    """Loads (T, 2048) feature arrays produced by extract_features.py."""

    def __init__(self, features_dir: Path, split: str) -> None:
        self.split_dir = features_dir / split
        index_path = features_dir / f"{split}_index.csv"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing feature index for split '{split}': {index_path}. "
                f"Run extract_features.py first."
            )
        df = pd.read_csv(index_path)
        self.items: list[tuple[Path, int]] = []
        for _, row in df.iterrows():
            p = self.split_dir / f"{row['video_id']}.npy"
            if not p.exists():
                raise FileNotFoundError(f"Cached feature missing: {p}")
            self.items.append((p, int(row["label"])))

        labels = [lbl for _, lbl in self.items]
        self.class_counts = {0: labels.count(0), 1: labels.count(1)}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[idx]
        feats = np.load(path)  # (T, 2048) float32
        return torch.from_numpy(feats), torch.tensor(float(label), dtype=torch.float32)


def make_balanced_sampler(dataset: CachedFeatureDataset) -> WeightedRandomSampler:
    counts = dataset.class_counts
    weights_per_class = {c: 1.0 / max(n, 1) for c, n in counts.items()}
    sample_weights = [weights_per_class[lbl] for _, lbl in dataset.items]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def epoch_loop(
    head: LSTMTemporalClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    head.train(is_train)
    total_loss = 0.0
    n = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)
            logits = head(feats)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * feats.size(0)
            n += feats.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.long().cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_real = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_fake = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_real": f1_real,
        "f1_fake": f1_fake,
    }


def assemble_full_checkpoint(
    head: LSTMTemporalClassifier,
    args: argparse.Namespace,
    history: dict,
    best_metric: float,
    epoch: int,
    out_path: Path,
) -> None:
    """Build a HybridCNNLSTM with ImageNet backbone + trained head, save full state_dict.

    This matches what evaluate_hybrid.py expects so it can load the file unchanged.
    """
    device = torch.device("cpu")  # checkpoint files are device-agnostic
    full_model = HybridCNNLSTM(
        pretrained=True,  # same IMAGENET1K_V2 weights used during feature extraction
        freeze_backbone=False,
        trainable_backbone_layers=4,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        bidirectional=not args.no_bidirectional,
        dropout=args.dropout,
    ).to(device)
    # Replace temporal head with trained one
    full_model.temporal.load_state_dict(head.state_dict())
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": full_model.state_dict(),
            "best_val_macro_f1": best_metric,
            "history": history,
            "config": vars(args),
            "notes": (
                "Backbone is frozen ImageNet ResNet-50 IMAGENET1K_V2; only the "
                "LSTM head was trained on cached features. Compatible with "
                "evaluate_hybrid.py."
            ),
        },
        out_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(args.project_root).resolve()
    features_dir = (
        Path(args.features_dir).resolve()
        if args.features_dir
        else (project_root / "data" / "features")
    )
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Features dir not found: {features_dir}. Run extract_features.py first."
        )

    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_set = CachedFeatureDataset(features_dir, "train")
    val_set = CachedFeatureDataset(features_dir, "val")
    print(f"Train: {len(train_set)} videos, classes={train_set.class_counts}")
    print(f"Val  : {len(val_set)} videos, classes={val_set.class_counts}")

    sampler = make_balanced_sampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    head = LSTMTemporalClassifier(
        feature_dim=2048,
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_num_layers,
        bidirectional=not args.no_bidirectional,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"LSTM head parameters: {n_params/1e6:.2f}M")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_metric = -1.0
    best_state = None
    epochs_without_improvement = 0
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [], "train_macro_f1": [],
        "val_loss": [], "val_acc": [], "val_macro_f1": [],
        "val_f1_real": [], "val_f1_fake": [], "lr": [],
    }

    head_path = models_dir / args.head_only_checkpoint
    full_path = models_dir / args.checkpoint_name

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = epoch_loop(head, train_loader, criterion, optimizer, device)
        val_metrics = epoch_loop(head, val_loader, criterion, None, device)
        scheduler.step(val_metrics["macro_f1"])
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_f1_real"].append(val_metrics["f1_real"])
        history["val_f1_fake"].append(val_metrics["f1_fake"])
        history["lr"].append(lr)

        elapsed = time.time() - t0
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} "
            f"mF1={train_metrics['macro_f1']:.3f} | "
            f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f} "
            f"mF1={val_metrics['macro_f1']:.3f} "
            f"(real={val_metrics['f1_real']:.3f} fake={val_metrics['f1_fake']:.3f}) "
            f"| lr={lr:.5f} | {elapsed:.1f}s"
        )

        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            torch.save(
                {
                    "epoch": epoch,
                    "head_state_dict": best_state,
                    "best_val_macro_f1": best_metric,
                    "history": history,
                    "config": vars(args),
                },
                head_path,
            )
            epochs_without_improvement = 0
            print(f"  saved best head -> {head_path} (val macro-F1={best_metric:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stop: no val macro-F1 improvement for "
                      f"{args.early_stop_patience} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training produced no best checkpoint.")
    head.load_state_dict(best_state)

    assemble_full_checkpoint(
        head=head,
        args=args,
        history=history,
        best_metric=best_metric,
        epoch=len(history["val_loss"]),
        out_path=full_path,
    )
    print(f"Saved full hybrid checkpoint -> {full_path}")
    print(f"Best validation macro-F1: {best_metric:.4f}")

    history_path = outputs_dir / "train_history_hybrid_v2.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    print(f"History -> {history_path}")


if __name__ == "__main__":
    main()
