"""Centralised settings for the deepfake-detection web app."""

from __future__ import annotations

import json
import os
from pathlib import Path


class Config:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    APP_ROOT: Path = Path(__file__).resolve().parent

    # Model selection: "cnn" (frame-level baseline, deployed) or "hybrid".
    MODEL_KIND: str = os.environ.get("APP_MODEL_KIND", "cnn").lower()

    # CNN baseline (deployed) — test accuracy 77.58% / ROC-AUC 0.766 at thr 0.75.
    CNN_MODEL_PATH: Path = PROJECT_ROOT / "models" / "cnn_baseline_best.pth"
    CNN_DROPOUT: float = 0.4
    CNN_TRAINABLE_BACKBONE_LAYERS: int = 1

    # Hybrid (kept for future use after a properly fine-tuned checkpoint exists).
    HYBRID_MODEL_PATH: Path = PROJECT_ROOT / "models" / "hybrid_best.pth"
    LSTM_HIDDEN_SIZE: int = 64
    LSTM_NUM_LAYERS: int = 1
    LSTM_BIDIRECTIONAL: bool = True
    LSTM_DROPOUT: float = 0.6

    # Preprocessing
    NUM_FRAMES: int = 32
    IMAGE_SIZE: int = 224

    # Decision threshold; will be overridden by outputs/threshold_<kind>.json
    # or outputs/threshold.json if either exists.
    DEFAULT_THRESHOLD_BY_KIND: dict[str, float] = {
        "cnn": 0.75,     # tuned on val for macro-F1
        "hybrid": 0.50,
    }

    # Upload
    UPLOAD_DIR: Path = APP_ROOT / "static" / "uploads"
    LOG_DIR: Path = APP_ROOT / "logs"
    MAX_CONTENT_LENGTH: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: set[str] = {"mp4", "avi", "mov", "mkv", "webm"}

    # Server
    HOST: str = os.environ.get("APP_HOST", "127.0.0.1")
    PORT: int = int(os.environ.get("APP_PORT", "5000"))
    DEBUG: bool = os.environ.get("APP_DEBUG", "0") == "1"
    SECRET_KEY: str = os.environ.get(
        "APP_SECRET_KEY",
        "dev-only-do-not-use-in-prod-9f8b2a3c4d5e6f7a",
    )

    # Inference safety
    INFERENCE_TIMEOUT_S: float = 120.0


def load_threshold(kind: str, default: float | None = None) -> float:
    """Resolve threshold: per-kind json > generic json > Config default > 0.5."""
    candidates = [
        Config.PROJECT_ROOT / "outputs" / f"threshold_{kind}.json",
        Config.PROJECT_ROOT / "outputs" / "threshold.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "threshold" in data:
                return float(data["threshold"])
        except Exception:
            continue
    if default is not None:
        return default
    return Config.DEFAULT_THRESHOLD_BY_KIND.get(kind, 0.5)


def predictor_kwargs(kind: str) -> dict:
    """Return the kwargs needed to construct the configured predictor."""
    kind = kind.lower()
    if kind == "cnn":
        return {
            "checkpoint_path": Config.CNN_MODEL_PATH,
            "dropout": Config.CNN_DROPOUT,
            "trainable_backbone_layers": Config.CNN_TRAINABLE_BACKBONE_LAYERS,
        }
    if kind == "hybrid":
        return {
            "checkpoint_path": Config.HYBRID_MODEL_PATH,
            "lstm_hidden_size": Config.LSTM_HIDDEN_SIZE,
            "lstm_num_layers": Config.LSTM_NUM_LAYERS,
            "bidirectional": Config.LSTM_BIDIRECTIONAL,
            "dropout": Config.LSTM_DROPOUT,
        }
    raise ValueError(f"Unknown MODEL_KIND={kind!r}")
