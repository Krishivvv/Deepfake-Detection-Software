"""Forward-pass smoke tests on tiny batches (CPU, no pretrained download)."""

from __future__ import annotations

import torch

from src.models.lstm_temporal import LSTMTemporalClassifier
from src.models.resnet_classifier import DeepfakeClassifier


def test_cnn_forward_shape():
    model = DeepfakeClassifier(pretrained=False, trainable_backbone_layers=0)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_cnn_predict_proba_in_unit_interval():
    model = DeepfakeClassifier(pretrained=False, trainable_backbone_layers=0)
    probs = model.predict_proba(torch.randn(4, 3, 224, 224))
    assert probs.shape == (4,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_lstm_head_forward_shape():
    head = LSTMTemporalClassifier(feature_dim=2048, hidden_size=128,
                                  num_layers=1, bidirectional=True)
    head.eval()
    with torch.no_grad():
        logits = head(torch.randn(3, 8, 2048))  # (B, T, feat)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_backbone_as_feature_extractor_emits_2048d():
    """fc -> Identity turns the backbone into a 2048-d extractor (hybrid_v3 path)."""
    import torch.nn as nn

    model = DeepfakeClassifier(pretrained=False, trainable_backbone_layers=0)
    model.backbone.fc = nn.Identity()
    model.eval()
    with torch.no_grad():
        feats = model.backbone(torch.randn(2, 3, 224, 224))
    assert feats.shape == (2, 2048)
