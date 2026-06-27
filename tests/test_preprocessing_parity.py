"""
Train/inference preprocessing parity.

The model only ever sees correctly-normalized tensors if training and inference
use identical eval transforms. These tests lock that contract: the ImageNet
constants and image size in the dataset module must match config.yaml, and the
eval transform must reproduce a hand-built normalize pipeline bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import load_config
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms


def test_constants_match_config():
    cfg = load_config()
    assert cfg.imagenet_mean == IMAGENET_MEAN
    assert cfg.imagenet_std == IMAGENET_STD
    assert cfg.image_size == 224


def test_eval_transform_matches_manual_pipeline():
    cfg = load_config()
    img = Image.fromarray(
        (np.random.default_rng(1).random((80, 120, 3)) * 255).astype("uint8")
    )

    produced = get_eval_transforms(image_size=cfg.image_size)(img)
    manual = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])(img)

    assert produced.shape == (3, cfg.image_size, cfg.image_size)
    assert torch.allclose(produced, manual, atol=1e-6)


def test_eval_transform_has_no_augmentation():
    """Eval transform must be deterministic (no flip/rotation/jitter)."""
    img = Image.fromarray(
        (np.random.default_rng(2).random((64, 64, 3)) * 255).astype("uint8")
    )
    t = get_eval_transforms(image_size=224)
    assert torch.equal(t(img), t(img))
