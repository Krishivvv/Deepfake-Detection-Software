"""
Cache frozen ResNet-50 (ImageNet) features for every video in train/val/test.

CPU-only end-to-end training of HybridCNNLSTM is impractical, so we run the
backbone once and persist per-frame features. Phase 2 (train_hybrid_v2.py)
then trains only the temporal head on these cached tensors.

Output layout:
    data/features/<split>/<video_id>.npy   # float32, shape (T, 2048)
    data/features/<split>_index.csv        # video_id, label, n_frames
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import models
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.video_dataset import VideoSequenceDataset  # noqa: E402


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache ResNet-50 features per video.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-frames", type=int, default=32,
                        help="Frames per backbone forward call (one video at a time).")
    parser.add_argument("--max-videos", type=int, default=0,
                        help="If >0, only process this many videos per split (smoke test).")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS),
                        choices=list(SPLITS))
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if cached .npy already exists.")
    return parser.parse_args()


def load_backbone(device: torch.device) -> torch.nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    backbone = models.resnet50(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device)


def video_id_from_csv_path(raw: str) -> str:
    """Convert 'fake/neuraltextures/241_210' -> 'fake__neuraltextures__241_210'."""
    return raw.strip().replace("\\", "/").strip("/").replace("/", "__")


def extract_split(
    split: str,
    backbone: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    project_root = Path(args.project_root).resolve()
    csv_path = project_root / "data" / "splits" / f"{split}.csv"
    out_dir = project_root / "data" / "features" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = project_root / "data" / "features" / f"{split}_index.csv"

    dataset = VideoSequenceDataset(
        csv_path=csv_path,
        project_root=project_root,
        split="val",  # eval transforms = no augmentation -> deterministic cache
        num_frames=args.num_frames,
        image_size=args.image_size,
    )

    import pandas as pd
    df = pd.read_csv(csv_path)
    path_col = [c for c in df.columns if c.lower() in ("video_path", "path", "folder", "video", "frame_path")][0]
    label_col = [c for c in df.columns if c.lower() in ("label", "target", "class")][0]
    raw_paths = df[path_col].tolist()
    raw_labels = df[label_col].tolist()

    if len(raw_paths) != len(dataset):
        print(f"[{split}] WARNING: csv rows={len(raw_paths)} dataset items={len(dataset)} "
              f"(skipped={dataset.skipped_rows}); index will track dataset order.")

    n = len(dataset)
    if args.max_videos and args.max_videos > 0:
        n = min(n, args.max_videos)

    rows: list[tuple[str, int, int]] = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(n), desc=f"extract[{split}]"):
            clip, label = dataset[i]              # (T, 3, H, W), scalar tensor
            vid = video_id_from_csv_path(raw_paths[i])
            out_path = out_dir / f"{vid}.npy"
            if out_path.exists() and not args.overwrite:
                arr = np.load(out_path, mmap_mode="r")
                rows.append((vid, int(label.item()), int(arr.shape[0])))
                continue

            frames = clip.to(device, non_blocking=True)
            feats_chunks: list[torch.Tensor] = []
            for start in range(0, frames.shape[0], args.batch_frames):
                chunk = frames[start : start + args.batch_frames]
                feats_chunks.append(backbone(chunk).cpu())
            feats = torch.cat(feats_chunks, dim=0).numpy().astype(np.float32)

            if feats.ndim != 2 or feats.shape[1] != 2048:
                raise RuntimeError(
                    f"Unexpected feature shape {feats.shape} for {vid} "
                    f"(expected (T, 2048))."
                )
            np.save(out_path, feats)
            rows.append((vid, int(label.item()), int(feats.shape[0])))

    with index_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["video_id", "label", "n_frames"])
        w.writerows(rows)

    counts = {0: sum(1 for _, l, _ in rows if l == 0),
              1: sum(1 for _, l, _ in rows if l == 1)}
    elapsed = time.time() - t0
    print(f"[{split}] cached={len(rows)} class_counts={counts} "
          f"elapsed={elapsed:.1f}s -> {out_dir}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    backbone = load_backbone(device)
    print("Backbone loaded (ResNet-50 IMAGENET1K_V2, fc=Identity, frozen).")

    for split in args.splits:
        extract_split(split, backbone, device, args)

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
