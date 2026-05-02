"""
Model wrapper for deepfake-detection inference.

Two predictor implementations are provided:

* ``CNNFramePredictor`` — wraps the frame-level CNN (``DeepfakeClassifier``)
  and aggregates per-frame probabilities into a single video-level decision.
  This is the **deployed** model (test accuracy 77.58%, ROC-AUC 0.7657 at
  threshold 0.75).

* ``HybridPredictor`` — wraps the end-to-end ``HybridCNNLSTM``. Kept for
  backwards compatibility / future use once a properly fine-tuned hybrid
  checkpoint exists.

The Flask app picks one based on ``Config.MODEL_KIND`` ("cnn" or "hybrid").
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn  # noqa: E402

from src.models.hybrid_model import HybridCNNLSTM  # noqa: E402
from src.models.lstm_temporal import LSTMTemporalClassifier  # noqa: E402
from src.models.resnet_classifier import DeepfakeClassifier  # noqa: E402

log = logging.getLogger("app.predictor")


class ModelLoadError(Exception):
    """Raised when the saved checkpoint cannot be loaded."""


class PredictionError(Exception):
    """Raised when forward pass fails or produces non-finite outputs."""


class CNNFramePredictor:
    """Frame-level CNN baseline; video-level decision = mean of frame probs."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        dropout: float = 0.4,
        trainable_backbone_layers: int = 1,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not self.checkpoint_path.exists():
            raise ModelLoadError(
                f"Model checkpoint not found: {self.checkpoint_path}."
            )
        try:
            self.model = DeepfakeClassifier(
                pretrained=False,
                dropout=dropout,
                trainable_backbone_layers=trainable_backbone_layers,
            ).to(self.device)
            ck = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ck["model_state_dict"] if "model_state_dict" in ck else ck
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.config_meta = ck.get("config", {}) if isinstance(ck, dict) else {}
            self.notes = (
                "Frame-level ResNet-50 CNN baseline. Video-level decision "
                "is the mean of per-frame fake probabilities."
            )
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load CNN checkpoint")
            raise ModelLoadError(f"Could not load model checkpoint: {exc!s}") from exc

        log.info("Loaded CNNFramePredictor (device=%s) from %s",
                 self.device, self.checkpoint_path)

    @torch.no_grad()
    def predict(self, clip: torch.Tensor, threshold: float = 0.5) -> dict:
        """``clip`` shape: (1, T, 3, H, W). Returns prediction dict.

        Per-frame probabilities are averaged to a single video-level fake
        probability; that is compared to ``threshold``.
        """
        if clip.dim() != 5:
            raise PredictionError(
                f"Expected 5D clip tensor (B,T,C,H,W), got shape {tuple(clip.shape)}"
            )
        clip = clip.to(self.device, non_blocking=True)
        b, t, c, h, w = clip.shape
        flat = clip.view(b * t, c, h, w)

        t0 = time.time()
        try:
            logits = self.model(flat)            # (B*T,)
        except RuntimeError as exc:
            log.exception("Model forward pass failed")
            raise PredictionError(
                "Model inference failed. Try a shorter or simpler video."
            ) from exc
        elapsed = time.time() - t0

        if not torch.isfinite(logits).all():
            raise PredictionError(
                "Model produced non-finite outputs (NaN/Inf). Try a different video."
            )

        frame_probs = torch.sigmoid(logits).view(b, t)        # (B, T)
        prob_fake = float(frame_probs.mean(dim=1).item())     # B=1
        is_fake = prob_fake >= threshold
        confidence = prob_fake if is_fake else (1.0 - prob_fake)

        # Frame-level diagnostics for the UI / logs.
        frame_list = frame_probs[0].cpu().tolist()
        frames_above = int(sum(1 for p in frame_list if p >= threshold))

        return {
            "label": "FAKE" if is_fake else "REAL",
            "is_fake": bool(is_fake),
            "probability_fake": prob_fake,
            "probability_real": 1.0 - prob_fake,
            "confidence": confidence,
            "threshold": float(threshold),
            "inference_seconds": elapsed,
            "device": str(self.device),
            "frames_evaluated": int(t),
            "frames_above_threshold": frames_above,
            "frame_probabilities": [round(p, 4) for p in frame_list],
        }


class HybridPredictor:
    """End-to-end CNN-LSTM hybrid; kept for compatibility with future checkpoints."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not self.checkpoint_path.exists():
            raise ModelLoadError(
                f"Model checkpoint not found: {self.checkpoint_path}."
            )
        try:
            self.model = HybridCNNLSTM(
                pretrained=False,
                freeze_backbone=False,
                trainable_backbone_layers=4,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            ).to(self.device)
            ck = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ck["model_state_dict"] if "model_state_dict" in ck else ck
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.config_meta = ck.get("config", {}) if isinstance(ck, dict) else {}
            self.notes = ck.get("notes", "") if isinstance(ck, dict) else ""
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load Hybrid checkpoint")
            raise ModelLoadError(f"Could not load model checkpoint: {exc!s}") from exc

        log.info("Loaded HybridPredictor (device=%s) from %s",
                 self.device, self.checkpoint_path)

    @torch.no_grad()
    def predict(self, clip: torch.Tensor, threshold: float = 0.5) -> dict:
        if clip.dim() != 5:
            raise PredictionError(
                f"Expected 5D clip tensor (B,T,C,H,W), got shape {tuple(clip.shape)}"
            )
        clip = clip.to(self.device, non_blocking=True)
        t0 = time.time()
        try:
            logits = self.model(clip)
        except RuntimeError as exc:
            log.exception("Model forward pass failed")
            raise PredictionError(
                "Model inference failed. Try a shorter or simpler video."
            ) from exc
        elapsed = time.time() - t0

        if not torch.isfinite(logits).all():
            raise PredictionError(
                "Model produced non-finite outputs (NaN/Inf). Try a different video."
            )

        prob_fake = float(torch.sigmoid(logits).item())
        is_fake = prob_fake >= threshold
        confidence = prob_fake if is_fake else (1.0 - prob_fake)
        return {
            "label": "FAKE" if is_fake else "REAL",
            "is_fake": bool(is_fake),
            "probability_fake": prob_fake,
            "probability_real": 1.0 - prob_fake,
            "confidence": confidence,
            "threshold": float(threshold),
            "inference_seconds": elapsed,
            "device": str(self.device),
        }


class HybridV3Predictor:
    """Composite hybrid: CNN-baseline backbone (frozen at inference) + trained LSTM head.

    The backbone is the trained ResNet-50 from ``cnn_baseline_best.pth`` with
    its classification head replaced by ``Identity`` so it emits 2048-d
    per-frame features. The temporal head is an ``LSTMTemporalClassifier``
    trained by ``train_hybrid_v2.py --features-dir data/features_cnn`` and
    saved to ``hybrid_v3_head.pth``. This composite reproduces exactly the
    feature pipeline used at training time.

    Why not load a single ``HybridCNNLSTM`` checkpoint? Because
    ``train_hybrid_v2.py``'s ``assemble_full_checkpoint`` plugs in fresh
    ImageNet weights for the backbone — if the head was trained on
    CNN-baseline features (as v3 was), the assembled file would have a
    backbone/head mismatch.
    """

    def __init__(
        self,
        cnn_backbone_checkpoint: str | Path,
        head_checkpoint: str | Path,
        cnn_dropout: float = 0.4,
        cnn_trainable_backbone_layers: int = 1,
        device: Optional[str] = None,
    ) -> None:
        self.cnn_path = Path(cnn_backbone_checkpoint)
        self.head_path = Path(head_checkpoint)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        for p in (self.cnn_path, self.head_path):
            if not p.exists():
                raise ModelLoadError(f"Required checkpoint missing: {p}")
        try:
            cnn = DeepfakeClassifier(
                pretrained=False, dropout=cnn_dropout,
                trainable_backbone_layers=cnn_trainable_backbone_layers,
            ).to(self.device)
            cnn_ck = torch.load(self.cnn_path, map_location=self.device, weights_only=False)
            cnn_state = cnn_ck["model_state_dict"] if "model_state_dict" in cnn_ck else cnn_ck
            cnn.load_state_dict(cnn_state)
            cnn.backbone.fc = nn.Identity()
            cnn.eval()
            for param in cnn.parameters():
                param.requires_grad = False
            self.backbone = cnn.backbone

            head_ck = torch.load(self.head_path, map_location=self.device, weights_only=False)
            head_cfg = head_ck.get("config", {}) if isinstance(head_ck, dict) else {}
            self.head = LSTMTemporalClassifier(
                feature_dim=2048,
                hidden_size=int(head_cfg.get("lstm_hidden_size", 128)),
                num_layers=int(head_cfg.get("lstm_num_layers", 1)),
                bidirectional=not bool(head_cfg.get("no_bidirectional", False)),
                dropout=float(head_cfg.get("dropout", 0.5)),
            ).to(self.device)
            head_state = head_ck["head_state_dict"] if "head_state_dict" in head_ck \
                else head_ck.get("model_state_dict", head_ck)
            self.head.load_state_dict(head_state)
            self.head.eval()

            self.config_meta = head_cfg
            self.notes = (
                "Composite predictor: trained CNN-baseline ResNet-50 backbone "
                "(fc → Identity) + LSTM head trained on CNN-baseline features. "
                "Test accuracy 82.0 %, ROC-AUC 0.870."
            )
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load HybridV3 checkpoints")
            raise ModelLoadError(f"Could not load hybrid v3 checkpoints: {exc!s}") from exc

        log.info(
            "Loaded HybridV3Predictor (device=%s) backbone=%s head=%s",
            self.device, self.cnn_path, self.head_path,
        )

    @torch.no_grad()
    def predict(self, clip: torch.Tensor, threshold: float = 0.575) -> dict:
        if clip.dim() != 5:
            raise PredictionError(
                f"Expected 5D clip tensor (B,T,C,H,W), got shape {tuple(clip.shape)}"
            )
        clip = clip.to(self.device, non_blocking=True)
        b, t, c, h, w = clip.shape
        flat = clip.view(b * t, c, h, w)

        t0 = time.time()
        try:
            features = self.backbone(flat)        # (B*T, 2048)
            features = features.view(b, t, -1)    # (B, T, 2048)
            logits = self.head(features)          # (B,)
        except RuntimeError as exc:
            log.exception("Hybrid v3 forward pass failed")
            raise PredictionError(
                "Model inference failed. Try a shorter or simpler video."
            ) from exc
        elapsed = time.time() - t0

        if not torch.isfinite(logits).all():
            raise PredictionError(
                "Model produced non-finite outputs (NaN/Inf). Try a different video."
            )

        prob_fake = float(torch.sigmoid(logits).item())
        is_fake = prob_fake >= threshold
        confidence = prob_fake if is_fake else (1.0 - prob_fake)

        return {
            "label": "FAKE" if is_fake else "REAL",
            "is_fake": bool(is_fake),
            "probability_fake": prob_fake,
            "probability_real": 1.0 - prob_fake,
            "confidence": confidence,
            "threshold": float(threshold),
            "inference_seconds": elapsed,
            "device": str(self.device),
        }


def build_predictor(kind: str, **kwargs):
    """Factory returning the appropriate predictor."""
    kind = kind.lower().strip()
    if kind == "cnn":
        return CNNFramePredictor(**kwargs)
    if kind == "hybrid":
        return HybridPredictor(**kwargs)
    if kind in ("hybrid_v3", "hybridv3"):
        return HybridV3Predictor(**kwargs)
    raise ValueError(f"Unknown MODEL_KIND='{kind}'. Use 'cnn', 'hybrid', or 'hybrid_v3'.")
