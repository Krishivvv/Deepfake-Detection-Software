"""Shared pytest fixtures and path setup for the Veridex test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    """Build a self-contained frame-level dataset (no real data/weights needed).

    Layout:
        <tmp>/data/processed/real/clip_a/frame_00.jpg
        <tmp>/data/processed/fake/clip_b/frame_00.jpg
    plus a split CSV referencing the two clip folders. Returns (project_root, csv).
    """
    root = tmp_path
    rng = np.random.default_rng(0)

    def _make_clip(rel: str, n: int = 3) -> Path:
        d = root / "data" / "processed" / rel
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"frame_{i:02d}.jpg")
        return d

    _make_clip("real/clip_a")
    _make_clip("fake/clip_b")

    csv_path = root / "data" / "splits" / "train.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "video_path,label\n"
        "data/processed/real/clip_a,real\n"
        "data/processed/fake/clip_b,1\n",
        encoding="utf-8",
    )
    return root, csv_path
