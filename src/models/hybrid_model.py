"""
End-to-end CNN-LSTM hybrid for video-level deepfake detection.

The CNN backbone (ResNet-50, ImageNet pretrained) extracts a feature vector
per frame; a BiLSTM aggregates the sequence and emits a single video logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .lstm_temporal import LSTMTemporalClassifier


class HybridCNNLSTM(nn.Module):
    """
    Parameters
    ----------
    pretrained:
        If True, load ImageNet weights for the ResNet-50 backbone.
    freeze_backbone:
        If True, freeze the entire CNN (pure feature extractor mode).
        If False (default for end-to-end), the CNN is trained jointly.
    trainable_backbone_layers:
        Only consulted when freeze_backbone is False but you still want to
        partially freeze. 0..4 — number of trailing ResNet stages to keep
        trainable. 4 means train the whole backbone.
    lstm_hidden_size, lstm_num_layers, bidirectional, dropout:
        Forwarded to LSTMTemporalClassifier.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        trainable_backbone_layers: int = 4,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 2048
        # Drop the classification head; keep up to global avg pool.
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self._configure_backbone_grads(
            freeze_all=freeze_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
        )

        self.temporal = LSTMTemporalClassifier(
            feature_dim=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def _configure_backbone_grads(
        self,
        freeze_all: bool,
        trainable_backbone_layers: int,
    ) -> None:
        if freeze_all:
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        n = max(0, min(4, int(trainable_backbone_layers)))
        if n == 4:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        # Partial unfreeze: freeze early stages, train trailing ones.
        for param in self.backbone.parameters():
            param.requires_grad = False
        stages = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]
        for stage in stages[-n:] if n > 0 else []:
            for param in stage.parameters():
                param.requires_grad = True

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips: (B, T, 3, H, W)
        returns logits: (B,)
        """
        if clips.dim() != 5:
            raise ValueError(
                f"Expected (B, T, 3, H, W), got shape {tuple(clips.shape)}"
            )
        b, t, c, h, w = clips.shape
        flat = clips.view(b * t, c, h, w)
        features = self.backbone(flat)         # (B*T, feature_dim)
        features = features.view(b, t, -1)     # (B, T, feature_dim)
        logits = self.temporal(features)       # (B,)
        return logits

    @staticmethod
    def _batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        return (preds == targets).float().mean().item()

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
    ) -> dict[str, float]:
        self.train()
        clips, labels = batch
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = self(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = self._batch_accuracy(logits.detach(), labels)
        return {"loss": loss.item(), "accuracy": acc}

    def val_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        device: torch.device | str,
    ) -> dict[str, float]:
        self.eval()
        clips, labels = batch
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        with torch.no_grad():
            logits = self(clips)
            loss = criterion(logits, labels)
            acc = self._batch_accuracy(logits, labels)
        return {"loss": loss.item(), "accuracy": acc}


def create_hybrid_model_and_optimizer(
    device: torch.device | str,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    trainable_backbone_layers: int = 4,
    lstm_hidden_size: int = 256,
    lstm_num_layers: int = 2,
    bidirectional: bool = True,
    dropout: float = 0.5,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> tuple[HybridCNNLSTM, nn.Module, torch.optim.Optimizer]:
    model = HybridCNNLSTM(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return model, criterion, optimizer
