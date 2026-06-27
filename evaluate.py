"""
Unified evaluation entry point for Veridex.

Replaces the former ``evaluate_cnn.py``, ``evaluate_hybrid.py`` and
``evaluate_hybrid_cached.py`` with a single parameterised CLI:

    python evaluate.py --model cnn          # frame-level ResNet-50 baseline
    python evaluate.py --model hybrid       # end-to-end CNN-LSTM (legacy)
    python evaluate.py --model hybrid_v3    # DEPLOYED: CNN backbone + LSTM head

All defaults (paths, thresholds, hyper-parameters) come from ``config.yaml``;
nothing is hard-coded to a machine. Reports are written to ``outputs/`` via
:mod:`src.evaluation.metrics`.

For ``hybrid_v3`` (cached-feature path) a val threshold sweep selects the
macro-F1-optimal threshold and the test set is reported at both that threshold
and 0.5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.evaluation.metrics import compute_binary_metrics, save_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Veridex evaluation.")
    parser.add_argument("--model", choices=["cnn", "hybrid", "hybrid_v3"],
                        default="hybrid_v3", help="Which model to evaluate.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Override the model checkpoint path.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold. Defaults to the config value; "
                             "for hybrid_v3 a val sweep picks it automatically.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# CNN frame-level evaluation
# --------------------------------------------------------------------------- #
def evaluate_cnn(cfg, args, device) -> None:
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from src.data.dataset import DeepfakeDataset
    from src.models.resnet_classifier import DeepfakeClassifier

    mcfg = cfg.model("cnn")
    ckpt = Path(args.checkpoint) if args.checkpoint else cfg.resolve(mcfg["checkpoint"])
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    threshold = args.threshold if args.threshold is not None else float(mcfg["threshold"])

    dataset = DeepfakeDataset(
        csv_path=cfg.dir("splits_dir") / "test.csv",
        project_root=cfg.project_root,
        split="test",
        image_size=cfg.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = DeepfakeClassifier(
        pretrained=False,
        dropout=float(mcfg["dropout"]),
        trainable_backbone_layers=int(mcfg["trainable_backbone_layers"]),
    ).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating CNN", leave=False):
            images = images.to(device, non_blocking=True)
            probs = torch.sigmoid(model(images))
            y_true.extend(labels.long().view(-1).cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    metrics = compute_binary_metrics(y_true, y_prob, threshold)
    print(f"[cnn] {metrics.summary_line()}")
    paths = save_metrics(metrics, y_true, y_prob, cfg.dir("outputs_dir"), "cnn",
                         "CNN BASELINE EVALUATION REPORT",
                         extra_header={"Checkpoint": str(ckpt)})
    _print_written(paths)


# --------------------------------------------------------------------------- #
# Hybrid (end-to-end CNN-LSTM) evaluation
# --------------------------------------------------------------------------- #
def evaluate_hybrid(cfg, args, device) -> None:
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from src.data.video_dataset import VideoSequenceDataset
    from src.models.hybrid_model import HybridCNNLSTM

    mcfg = cfg.model("hybrid")
    ckpt = Path(args.checkpoint) if args.checkpoint else cfg.resolve(mcfg["checkpoint"])
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    threshold = args.threshold if args.threshold is not None else float(mcfg["threshold"])

    dataset = VideoSequenceDataset(
        csv_path=cfg.dir("splits_dir") / "test.csv",
        project_root=cfg.project_root,
        split="test",
        num_frames=cfg.num_frames,
        image_size=cfg.image_size,
    )
    loader = DataLoader(dataset, batch_size=max(1, args.batch_size // 4), shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = HybridCNNLSTM(
        pretrained=False, freeze_backbone=False, trainable_backbone_layers=4,
        lstm_hidden_size=int(mcfg["lstm_hidden_size"]),
        lstm_num_layers=int(mcfg["lstm_num_layers"]),
        bidirectional=bool(mcfg["bidirectional"]),
        dropout=float(mcfg["dropout"]),
    ).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    model.eval()

    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="Evaluating hybrid", leave=False):
            clips = clips.to(device, non_blocking=True)
            probs = torch.sigmoid(model(clips))
            y_true.extend(labels.long().view(-1).cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    metrics = compute_binary_metrics(y_true, y_prob, threshold)
    print(f"[hybrid] {metrics.summary_line()}")
    paths = save_metrics(metrics, y_true, y_prob, cfg.dir("outputs_dir"), "hybrid",
                         "HYBRID CNN-LSTM EVALUATION REPORT",
                         extra_header={"Checkpoint": str(ckpt)})
    _print_written(paths)


# --------------------------------------------------------------------------- #
# Hybrid v3 (cached-feature) evaluation — the deployed model
# --------------------------------------------------------------------------- #
def _load_cached_features(features_dir: Path, split: str):
    import pandas as pd

    df = pd.read_csv(features_dir / f"{split}_index.csv")
    feats, labels = [], []
    for _, row in df.iterrows():
        feats.append(np.load(features_dir / split / f"{row['video_id']}.npy"))
        labels.append(int(row["label"]))
    return np.stack(feats, axis=0), np.array(labels)


def evaluate_hybrid_v3(cfg, args, device) -> None:
    from src.models.lstm_temporal import LSTMTemporalClassifier

    mcfg = cfg.model("hybrid_v3")
    head_ckpt = Path(args.checkpoint) if args.checkpoint else cfg.resolve(mcfg["head_checkpoint"])
    features_dir = cfg.resolve(mcfg["features_dir"])
    if not head_ckpt.exists():
        raise FileNotFoundError(f"Head checkpoint not found: {head_ckpt}")
    if not (features_dir / "test_index.csv").exists():
        raise FileNotFoundError(
            f"Cached features not found in {features_dir}. "
            f"Run: python extract_features.py --backbone cnn"
        )

    head_ck = torch.load(head_ckpt, map_location=device, weights_only=False)
    hcfg = head_ck.get("config", {}) if isinstance(head_ck, dict) else {}
    head = LSTMTemporalClassifier(
        feature_dim=2048,
        hidden_size=int(hcfg.get("lstm_hidden_size", mcfg["lstm_hidden_size"])),
        num_layers=int(hcfg.get("lstm_num_layers", mcfg["lstm_num_layers"])),
        bidirectional=not bool(hcfg.get("no_bidirectional", not mcfg["bidirectional"])),
        dropout=float(hcfg.get("dropout", mcfg["dropout"])),
    ).to(device)
    state = head_ck["head_state_dict"] if "head_state_dict" in head_ck \
        else head_ck.get("model_state_dict", head_ck)
    head.load_state_dict(state)
    head.eval()

    def probs_for(split: str):
        X, y = _load_cached_features(features_dir, split)
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(X), 32):
                batch = torch.from_numpy(X[start:start + 32]).to(device)
                out.extend(torch.sigmoid(head(batch)).cpu().tolist())
        return y, np.array(out)

    # Threshold selection on val (unless user forced one).
    yv, pv = probs_for("val")
    yt, pt = probs_for("test")
    if args.threshold is not None:
        best_thr = args.threshold
        val_mf1 = compute_binary_metrics(yv, pv, best_thr).macro_f1
    else:
        sweep = [(t, compute_binary_metrics(yv, pv, t).macro_f1)
                 for t in np.linspace(0.05, 0.95, 37)]
        best_thr, val_mf1 = max(sweep, key=lambda r: r[1])

    metrics = compute_binary_metrics(yt, pt, best_thr)
    print(f"[hybrid_v3] val-tuned thr={best_thr:.3f} (val macroF1={val_mf1:.4f})")
    print(f"[hybrid_v3] {metrics.summary_line()}")
    paths = save_metrics(metrics, yt, pt, cfg.dir("outputs_dir"), "hybrid",
                         "HYBRID CNN-LSTM EVALUATION REPORT (cached-features path)",
                         extra_header={"Head checkpoint": str(head_ckpt),
                                       "Val macro-F1": f"{val_mf1:.4f}",
                                       "Features dir": str(features_dir)})
    _print_written(paths)


def _print_written(paths: dict) -> None:
    print("Wrote:")
    for p in paths.values():
        print(f"  {p}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = _device()
    print(f"Device: {device} | model: {args.model}")
    {"cnn": evaluate_cnn, "hybrid": evaluate_hybrid, "hybrid_v3": evaluate_hybrid_v3}[args.model](
        cfg, args, device
    )


if __name__ == "__main__":
    main()
