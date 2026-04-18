"""
ResNet-50 transfer-learning classifier for binary deepfake detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class DeepfakeClassifier(nn.Module):
    """
    ResNet-50 model with a dropout + linear binary classification head.

    Parameters
    ----------
    pretrained:
        If True, initialize from ImageNet weights.
    dropout:
        Dropout probability before final linear output.
    trainable_backbone_layers:
        Number of last ResNet blocks to unfreeze (0-4).
        0 => freeze all backbone layers, train only classification head.
        1 => unfreeze layer4, 2 => layer3+layer4, etc.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.4,
        trainable_backbone_layers: int = 1,
    ) -> None:
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

        self.freeze_backbone(trainable_backbone_layers=trainable_backbone_layers)

    def freeze_backbone(self, trainable_backbone_layers: int = 1) -> None:
        trainable_backbone_layers = max(0, min(4, int(trainable_backbone_layers)))

        # Freeze everything first.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Always train classification head.
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        if trainable_backbone_layers == 0:
            return

        layers = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]
        for layer in layers[-trainable_backbone_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return logits.view(-1)

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
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = self(images)
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
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        with torch.no_grad():
            logits = self(images)
            loss = criterion(logits, labels)
            acc = self._batch_accuracy(logits, labels)

        return {"loss": loss.item(), "accuracy": acc}

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs >= threshold).long()


def create_model_and_optimizer(
    device: torch.device | str,
    pretrained: bool = True,
    dropout: float = 0.4,
    trainable_backbone_layers: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[DeepfakeClassifier, nn.Module, torch.optim.Optimizer]:
    model = DeepfakeClassifier(
        pretrained=pretrained,
        dropout=dropout,
        trainable_backbone_layers=trainable_backbone_layers,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return model, criterion, optimizer

