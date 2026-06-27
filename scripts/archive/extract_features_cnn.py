"""
Cache features from the trained CNN baseline (`cnn_baseline_best.pth`) per video.

This is the same idea as ``extract_features.py`` but uses the **trained**
CNN's ResNet-50 backbone instead of frozen ImageNet weights. The CNN
baseline learned deepfake-specific features during its own training, so its
2048-d global-avg-pool activations are far more discriminative for this
task than ImageNet activations.

Output layout:
    data/features_cnn/<split>/<video_id>.npy   # float32, shape (T, 2048)
    data/features_cnn/<split>_index.csv        # video_id, label, n_frames
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.video_dataset import VideoSequenceDataset  # noqa: E402
from src.models.resnet_classifier import DeepfakeClassifier  # noqa: E402

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache CNN-baseline features per video.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--cnn-checkpoint", type=str,
                        default=str(PROJECT_ROOT / "models" / "cnn_baseline_best.pth"))
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-frames", type=int, default=32)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS),
                        choices=list(SPLITS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-dir-name", type=str, default="features_cnn")
    return parser.parse_args()


def load_cnn_backbone(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load the trained CNN baseline and replace its classification head with Identity.

    What stays: full ResNet-50 backbone weights as fine-tuned during CNN training
    (last block was trainable). Result is a (2048-d) feature extractor whose
    representations carry deepfake-specific signal.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")
    model = DeepfakeClassifier(
        pretrained=False,
        dropout=0.4,
        trainable_backbone_layers=1,
    )
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(state_dict)
    # Strip the dropout+linear head: replace with Identity so backbone output
    # is the raw 2048-d global-avg-pool vector.
    model.backbone.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def video_id_from_csv_path(raw: str) -> str:
    return raw.strip().replace("\\", "/").strip("/").replace("/", "__")


def extract_split(
    split: str,
    backbone: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    project_root = Path(args.project_root).resolve()
    csv_path = project_root / "data" / "splits" / f"{split}.csv"
    out_dir = project_root / "data" / args.out_dir_name / split
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = project_root / "data" / args.out_dir_name / f"{split}_index.csv"

    dataset = VideoSequenceDataset(
        csv_path=csv_path,
        project_root=project_root,
        split="val",
        num_frames=args.num_frames,
        image_size=args.image_size,
    )

    import pandas as pd
    df = pd.read_csv(csv_path)
    path_col = [c for c in df.columns if c.lower() in (
        "video_path", "path", "folder", "video", "frame_path",
    )][0]
    raw_paths = df[path_col].tolist()

    n = len(dataset)
    if args.max_videos and args.max_videos > 0:
        n = min(n, args.max_videos)

    rows: list[tuple[str, int, int]] = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(n), desc=f"cnn-extract[{split}]"):
            clip, label = dataset[i]
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
                feats_chunks.append(backbone.backbone(chunk).cpu())
            feats = torch.cat(feats_chunks, dim=0).numpy().astype(np.float32)

            if feats.ndim != 2 or feats.shape[1] != 2048:
                raise RuntimeError(
                    f"Unexpected feature shape {feats.shape} for {vid}; "
                    f"expected (T, 2048)."
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

    backbone = load_cnn_backbone(Path(args.cnn_checkpoint), device)
    print(f"Loaded CNN backbone (fc -> Identity, frozen) from {args.cnn_checkpoint}")

    for split in args.splits:
        extract_split(split, backbone, device, args)

    print("CNN feature extraction complete.")


if __name__ == "__main__":
    main()
