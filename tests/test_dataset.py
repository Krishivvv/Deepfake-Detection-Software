"""Dataset loader: correct shapes, label encoding, and folder expansion."""

from __future__ import annotations

import torch

from src.data.dataset import DeepfakeDataset, _parse_label


def test_parse_label_variants():
    assert _parse_label("real") == 0
    assert _parse_label("FAKE") == 1
    assert _parse_label(0) == 0
    assert _parse_label("1") == 1


def test_dataset_shapes_and_labels(tiny_dataset):
    root, csv_path = tiny_dataset
    ds = DeepfakeDataset(csv_path=csv_path, project_root=root, split="test", image_size=224)

    # Two clips of 3 frames each -> 6 frame-level samples.
    assert len(ds) == 6
    assert ds.class_counts == {0: 3, 1: 3}

    image, label = ds[0]
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert label.item() in (0.0, 1.0)


def test_dataset_eval_transform_is_deterministic(tiny_dataset):
    root, csv_path = tiny_dataset
    ds = DeepfakeDataset(csv_path=csv_path, project_root=root, split="test", image_size=224)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)  # eval transform => no random augmentation
