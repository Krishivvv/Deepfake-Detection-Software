"""
Project configuration loader.

Reads ``config.yaml`` from the project root and exposes a small typed wrapper.
All directory paths are resolved RELATIVE to the project root, so the codebase
contains no machine-specific absolute paths and runs unchanged anywhere.

Usage
-----
    from src.config import load_config

    cfg = load_config()                       # uses <repo>/config.yaml
    model_cfg = cfg.model("hybrid_v3")        # dict of that model's settings
    ckpt = cfg.resolve(model_cfg["head_checkpoint"])   # absolute Path
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The project root is the directory that contains this file's parent ("src/").
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Thin wrapper around the parsed ``config.yaml`` mapping."""

    raw: dict[str, Any]
    project_root: Path = PROJECT_ROOT

    # -- generic access -----------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a config path against the project root (absolute paths pass through)."""
        p = Path(relative)
        return p if p.is_absolute() else (self.project_root / p)

    # -- convenience accessors ---------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.get("seed", default=42))

    @property
    def image_size(self) -> int:
        return int(self.get("preprocessing", "image_size", default=224))

    @property
    def num_frames(self) -> int:
        return int(self.get("preprocessing", "num_frames", default=32))

    @property
    def imagenet_mean(self) -> list[float]:
        return list(self.get("preprocessing", "imagenet_mean",
                             default=[0.485, 0.456, 0.406]))

    @property
    def imagenet_std(self) -> list[float]:
        return list(self.get("preprocessing", "imagenet_std",
                             default=[0.229, 0.224, 0.225]))

    def dir(self, name: str) -> Path:
        """Resolve one of the ``project.*`` directories to an absolute Path."""
        rel = self.get("project", name)
        if rel is None:
            raise KeyError(f"Unknown project directory: {name!r}")
        return self.resolve(rel)

    def model(self, kind: str) -> dict[str, Any]:
        cfg = self.get("models", kind)
        if cfg is None:
            known = list((self.get("models") or {}).keys())
            raise KeyError(f"Unknown model kind {kind!r}. Known: {known}")
        return dict(cfg)


def load_config(path: str | Path | None = None) -> Config:
    """Load and parse ``config.yaml`` (defaults to ``<repo>/config.yaml``)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    return Config(raw=raw)
