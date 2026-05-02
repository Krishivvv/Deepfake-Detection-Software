"""
PyTorch dataset and dataloader utilities for frame-level deepfake classification.

Supports both CSV formats:
1. Frame-level rows (path points to an image file)
2. Video-level rows (path points to a folder of extracted frames)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _parse_label(raw_label: object) -> int:
    if isinstance(raw_label, str):
        normalized = raw_label.strip().lower()
        if normalized in {"real", "0"}:
            return 0
        if normalized in {"fake", "1"}:
            return 1
        raise ValueError(f"Unsupported string label: {raw_label}")
    value = int(raw_label)
    if value not in (0, 1):
        raise ValueError(f"Label must be 0 or 1, got: {raw_label}")
    return value


def _get_column_name(columns: Iterable[str], candidates: tuple[str, ...]) -> str:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"CSV must include one of {candidates}. Found: {list(columns)}")


class DeepfakeDataset(Dataset):
    """
    Frame-level dataset built from split CSV files.

    Parameters
    ----------
    csv_path:
        Split CSV path (train/val/test).
    project_root:
        Repository root path.
    split:
        One of "train", "val", "test". Used to choose default transforms.
    transform:
        Optional custom transform. If None, split-based defaults are used.
    max_frames_per_video:
        Limit frames loaded from each video folder (useful for quick experiments).
    max_samples:
        Cap number of frame samples loaded from the CSV expansion.
    strict_paths:
        If True, raise when a CSV path cannot be resolved.
    """

    def __init__(
        self,
        csv_path: str | Path,
        project_root: str | Path,
        split: str = "train",
        transform: transforms.Compose | None = None,
        image_size: int = 224,
        max_frames_per_video: int | None = None,
        max_samples: int | None = None,
        strict_paths: bool = False,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.split = split.lower()
        self.max_frames_per_video = max_frames_per_video
        self.strict_paths = strict_paths
        self.transform = transform or (
            get_train_transforms(image_size=image_size)
            if self.split == "train"
            else get_eval_transforms(image_size=image_size)
        )

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        self.path_column = _get_column_name(
            df.columns,
            (
                "frame_path",
                "image_path",
                "path",
                "filepath",
                "file_path",
                "video_path",
            ),
        )
        self.label_column = _get_column_name(df.columns, ("label", "target", "class"))

        self.samples: list[tuple[Path, int]] = []
        skipped_rows = 0

        for _, row in df.iterrows():
            label = _parse_label(row[self.label_column])
            raw_path = str(row[self.path_column]).strip()
            resolved = self._resolve_raw_path(raw_path)

            row_samples = self._expand_path_to_samples(resolved, label)
            if not row_samples:
                skipped_rows += 1
                if self.strict_paths:
                    raise FileNotFoundError(
                        f"Could not resolve row path '{raw_path}' from {self.csv_path}"
                    )
                continue
            self.samples.extend(row_samples)

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise RuntimeError(
                f"No frame samples were built from CSV: {self.csv_path}. "
                "Check CSV paths and project_root."
            )

        labels = [label for _, label in self.samples]
        self.class_counts = {0: labels.count(0), 1: labels.count(1)}
        self.skipped_rows = skipped_rows

    def _resolve_raw_path(self, raw_path: str) -> Path:
        normalized = raw_path.replace("\\", "/").lstrip("./")
        candidate_paths = [
            Path(raw_path),
            self.project_root / normalized,
            self.project_root / "data" / "processed" / normalized,
        ]

        # Handle rows like "data/processed/real/.../frame_05.jpg"
        if normalized.startswith("data/processed/"):
            candidate_paths.append(self.project_root / normalized)

        # Deduplicate while preserving order
        deduped: list[Path] = []
        seen = set()
        for candidate in candidate_paths:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key not in seen:
                seen.add(key)
                deduped.append(candidate)

        for candidate in deduped:
            if candidate.exists():
                return candidate

        return deduped[0]

    def _expand_path_to_samples(self, path: Path, label: int) -> list[tuple[Path, int]]:
        if path.is_file():
            if path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                return [(path, label)]
            return []

        if path.is_dir():
            frame_paths = sorted(
                p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
            )
            if self.max_frames_per_video is not None:
                frame_paths = frame_paths[: self.max_frames_per_video]
            return [(frame_path, label) for frame_path in frame_paths]

        return []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        return image_tensor, label_tensor


def build_dataloaders(
    project_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    max_frames_per_video: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    project_root = Path(project_root)
    splits_dir = project_root / "data" / "splits"

    train_dataset = DeepfakeDataset(
        csv_path=splits_dir / "train.csv",
        project_root=project_root,
        split="train",
        image_size=image_size,
        max_frames_per_video=max_frames_per_video,
        max_samples=max_train_samples,
    )
    val_dataset = DeepfakeDataset(
        csv_path=splits_dir / "val.csv",
        project_root=project_root,
        split="val",
        image_size=image_size,
        max_frames_per_video=max_frames_per_video,
        max_samples=max_val_samples,
    )
    test_dataset = DeepfakeDataset(
        csv_path=splits_dir / "test.csv",
        project_root=project_root,
        split="test",
        image_size=image_size,
        max_frames_per_video=max_frames_per_video,
        max_samples=max_test_samples,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def get_dataloaders(*args, **kwargs):
    """Backward-compatible alias."""
    return build_dataloaders(*args, **kwargs)

