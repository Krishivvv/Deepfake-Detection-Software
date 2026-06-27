"""
Cache per-video ResNet-50 features for the CNN-LSTM training pipeline.

CPU-only end-to-end training of the hybrid model is impractical, so the
backbone is run once and per-frame features are persisted; the temporal head
is then trained on these cached tensors (see ``train_hybrid_v2.py``).

Consolidates the former ``extract_features.py`` (frozen ImageNet backbone) and
``extract_features_cnn.py`` (trained CNN-baseline backbone) into one CLI:

    python extract_features.py --backbone imagenet   # -> data/features/
    python extract_features.py --backbone cnn        # -> data/features_cnn/  (deployed)

The ``cnn`` backbone is the trained ResNet-50 from ``cnn_baseline_best.pth``
with its classification head replaced by Identity — its 2048-d activations are
deepfake-specific and far more discriminative than raw ImageNet activations,
which is why the deployed ``hybrid_v3`` head is trained on them.

Output layout:
    <out_dir>/<split>/<video_id>.npy   # float32, shape (T, 2048)
    <out_dir>/<split>_index.csv        # video_id, label, n_frames
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

from src.config import load_config  # noqa: E402
from src.data.video_dataset import VideoSequenceDataset  # noqa: E402

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache ResNet-50 features per video.")
    parser.add_argument("--backbone", choices=["imagenet", "cnn"], default="cnn",
                        help="imagenet = frozen ImageNet ResNet-50; "
                             "cnn = trained CNN-baseline backbone (deployed pipeline).")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--cnn-checkpoint", type=str, default="",
                        help="Override CNN-baseline checkpoint (only for --backbone cnn).")
    parser.add_argument("--out-dir", type=str, default="",
                        help="Override output directory (defaults from config).")
    parser.add_argument("--batch-frames", type=int, default=32)
    parser.add_argument("--max-videos", type=int, default=0,
                        help="If >0, only process this many videos per split (smoke test).")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_backbone(kind: str, cnn_ckpt: Path | None, device: torch.device) -> nn.Module:
    """Return a frozen 2048-d feature extractor (fc -> Identity)."""
    from torchvision import models

    from src.models.resnet_classifier import DeepfakeClassifier

    if kind == "imagenet":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
    else:  # cnn
        if cnn_ckpt is None or not cnn_ckpt.exists():
            raise FileNotFoundError(f"CNN checkpoint not found: {cnn_ckpt}")
        model = DeepfakeClassifier(pretrained=False, dropout=0.4, trainable_backbone_layers=1)
        ck = torch.load(cnn_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
        model.backbone.fc = nn.Identity()
        backbone = model.backbone

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device)


def _video_id(raw: str) -> str:
    return raw.strip().replace("\\", "/").strip("/").replace("/", "__")


def extract_split(split: str, backbone: nn.Module, device: torch.device,
                  cfg, out_root: Path, args: argparse.Namespace) -> None:
    import pandas as pd

    csv_path = cfg.dir("splits_dir") / f"{split}.csv"
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_root / f"{split}_index.csv"

    dataset = VideoSequenceDataset(
        csv_path=csv_path,
        project_root=cfg.project_root,
        split="val",  # eval transforms -> deterministic, no augmentation
        num_frames=cfg.num_frames,
        image_size=cfg.image_size,
    )

    df = pd.read_csv(csv_path)
    path_col = next(c for c in df.columns if c.lower() in
                    ("video_path", "path", "folder", "video", "frame_path"))
    raw_paths = df[path_col].tolist()

    n = len(dataset)
    if args.max_videos and args.max_videos > 0:
        n = min(n, args.max_videos)

    rows: list[tuple[str, int, int]] = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(n), desc=f"extract[{args.backbone}:{split}]"):
            clip, label = dataset[i]
            vid = _video_id(raw_paths[i])
            out_path = out_dir / f"{vid}.npy"
            if out_path.exists() and not args.overwrite:
                arr = np.load(out_path, mmap_mode="r")
                rows.append((vid, int(label.item()), int(arr.shape[0])))
                continue

            frames = clip.to(device, non_blocking=True)
            chunks: list[torch.Tensor] = []
            for start in range(0, frames.shape[0], args.batch_frames):
                chunks.append(backbone(frames[start:start + args.batch_frames]).cpu())
            feats = torch.cat(chunks, dim=0).numpy().astype(np.float32)
            if feats.ndim != 2 or feats.shape[1] != 2048:
                raise RuntimeError(f"Unexpected feature shape {feats.shape} for {vid}.")
            np.save(out_path, feats)
            rows.append((vid, int(label.item()), int(feats.shape[0])))

    with index_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["video_id", "label", "n_frames"])
        w.writerows(rows)

    counts = {0: sum(1 for _, lbl, _ in rows if lbl == 0),
              1: sum(1 for _, lbl, _ in rows if lbl == 1)}
    print(f"[{split}] cached={len(rows)} class_counts={counts} "
          f"elapsed={time.time() - t0:.1f}s -> {out_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | backbone: {args.backbone}")

    if args.out_dir:
        out_root = cfg.resolve(args.out_dir)
    else:
        out_root = cfg.dir("features_cnn_dir") if args.backbone == "cnn" else cfg.dir("features_dir")

    cnn_ckpt = None
    if args.backbone == "cnn":
        cnn_ckpt = (Path(args.cnn_checkpoint) if args.cnn_checkpoint
                    else cfg.resolve(cfg.model("cnn")["checkpoint"]))

    backbone = build_backbone(args.backbone, cnn_ckpt, device)
    print(f"Backbone ready (fc -> Identity, frozen). Output root: {out_root}")
    for split in args.splits:
        extract_split(split, backbone, device, cfg, out_root, args)
    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
