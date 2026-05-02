"""
Video-level dataset for CNN-LSTM hybrid deepfake detection.

Each item is a sequence of frames loaded in temporal order from one video
folder, returned as a tensor of shape (T, 3, H, W) together with its label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VALID_IMAGE_EXTENSIONS,
    _get_column_name,
    _parse_label,
)


def get_video_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmentations applied identically to every frame in a sequence.

    Randomness is sampled once per `__call__`, i.e. once per frame. If you
    want truly identical augmentation across all frames in a clip, wrap the
    per-frame call with a fixed seed upstream (not needed for ColorJitter-
    level noise which can help generalization).
    """
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


def get_video_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class VideoSequenceDataset(Dataset):
    """
    Loads all frames of a video as an ordered sequence.

    Parameters
    ----------
    csv_path:
        Path to a split CSV. Each row must point to a folder that contains
        frame images named so that sorted() yields temporal order (the
        existing preprocessing writes `frame_00.jpg ... frame_31.jpg`).
    project_root:
        Repository root; CSV paths are resolved relative to this root or
        relative to `<project_root>/data/processed`.
    split:
        "train", "val", or "test". Selects default transforms.
    num_frames:
        Number of frames per clip. If a video has more frames than this,
        they are uniformly subsampled; if fewer, the last frame is repeated
        (pad with edge) so the sequence length stays constant.
    transform:
        Optional override for the per-frame transform pipeline.
    strict_paths:
        If True, raise when a video folder cannot be resolved or is empty.
    """

    def __init__(
        self,
        csv_path: str | Path,
        project_root: str | Path,
        split: str = "train",
        num_frames: int = 32,
        image_size: int = 224,
        transform: transforms.Compose | None = None,
        strict_paths: bool = False,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.split = split.lower()
        self.num_frames = int(num_frames)
        self.strict_paths = strict_paths
        self.transform = transform or (
            get_video_train_transforms(image_size=image_size)
            if self.split == "train"
            else get_video_eval_transforms(image_size=image_size)
        )

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        self.path_column = _get_column_name(
            df.columns,
            ("video_path", "path", "folder", "video", "frame_path"),
        )
        self.label_column = _get_column_name(df.columns, ("label", "target", "class"))

        self.videos: list[tuple[list[Path], int]] = []
        skipped = 0

        for _, row in df.iterrows():
            label = _parse_label(row[self.label_column])
            raw_path = str(row[self.path_column]).strip()
            folder = self._resolve_folder(raw_path)
            frames = self._list_frames(folder)
            if not frames:
                skipped += 1
                if strict_paths:
                    raise FileNotFoundError(
                        f"Could not resolve/list frames for '{raw_path}' -> {folder}"
                    )
                continue
            self.videos.append((frames, label))

        if not self.videos:
            raise RuntimeError(
                f"No video sequences built from {self.csv_path}. "
                "Check CSV paths and project_root."
            )

        labels = [label for _, label in self.videos]
        self.class_counts = {0: labels.count(0), 1: labels.count(1)}
        self.skipped_rows = skipped

    def _resolve_folder(self, raw_path: str) -> Path:
        normalized = raw_path.replace("\\", "/").lstrip("./")
        candidates: list[Path] = [
            Path(raw_path),
            self.project_root / normalized,
            self.project_root / "data" / "processed" / normalized,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        # If rows point to a specific frame file, fall back to its parent dir.
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.parent
        return candidates[0]

    @staticmethod
    def _list_frames(folder: Path) -> list[Path]:
        if not folder.is_dir():
            return []
        frames = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        return frames

    def _select_indices(self, total: int) -> list[int]:
        """Return exactly `num_frames` indices in [0, total) in temporal order."""
        if total == self.num_frames:
            return list(range(total))
        if total > self.num_frames:
            # Uniformly spaced picks preserve temporal coverage.
            step = total / float(self.num_frames)
            return [min(total - 1, int(i * step)) for i in range(self.num_frames)]
        # total < num_frames: pad by repeating the last available frame.
        return list(range(total)) + [total - 1] * (self.num_frames - total)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_paths, label = self.videos[idx]
        indices = self._select_indices(len(frame_paths))

        tensors: list[torch.Tensor] = []
        for i in indices:
            image = Image.open(frame_paths[i]).convert("RGB")
            tensors.append(self.transform(image))

        sequence = torch.stack(tensors, dim=0)  # (T, 3, H, W)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        return sequence, label_tensor


def build_video_dataloaders(
    project_root: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    num_frames: int = 32,
    image_size: int = 224,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    project_root = Path(project_root)
    splits_dir = project_root / "data" / "splits"

    train_dataset = VideoSequenceDataset(
        csv_path=splits_dir / "train.csv",
        project_root=project_root,
        split="train",
        num_frames=num_frames,
        image_size=image_size,
    )
    val_dataset = VideoSequenceDataset(
        csv_path=splits_dir / "val.csv",
        project_root=project_root,
        split="val",
        num_frames=num_frames,
        image_size=image_size,
    )
    test_dataset = VideoSequenceDataset(
        csv_path=splits_dir / "test.csv",
        project_root=project_root,
        split="test",
        num_frames=num_frames,
        image_size=image_size,
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
