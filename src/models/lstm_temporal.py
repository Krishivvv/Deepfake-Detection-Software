"""
Bidirectional LSTM that turns a sequence of per-frame feature vectors into
a single video-level logit for binary deepfake classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMTemporalClassifier(nn.Module):
    """
    Parameters
    ----------
    feature_dim:
        Dimensionality of the per-frame features feeding the LSTM
        (e.g. 2048 for ResNet-50 global-pool output).
    hidden_size:
        LSTM hidden size per direction.
    num_layers:
        Stacked LSTM layers.
    bidirectional:
        If True, use a BiLSTM (doubles the effective hidden size seen by
        the classifier head).
    dropout:
        Dropout between LSTM layers (PyTorch applies it only if num_layers > 1)
        and before the final linear projection.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        classifier_in = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_in, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, T, feature_dim)
        returns : (B,) raw logits.
        """
        outputs, _ = self.lstm(features)  # (B, T, H * dirs)
        pooled = outputs.mean(dim=1)       # temporal mean-pool for stability
        logits = self.classifier(pooled)   # (B, 1)
        return logits.view(-1)
