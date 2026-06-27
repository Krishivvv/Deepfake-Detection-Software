"""
Download the deployed model weights from the Hugging Face Hub into ``models/``.

Run at container startup (see the backend Dockerfile) so the Flask app finds
``models/cnn_baseline_best.pth`` and ``models/hybrid_v3_head.pth`` exactly where
``app/config.py`` expects them — without ever committing weights to git.

Configured via env vars (sensible defaults):
    VERIDEX_HF_REPO       default "krishivvv/veridex-deepfake"
    VERIDEX_BACKBONE_FILE default "cnn_baseline_best.pth"
    VERIDEX_HEAD_FILE     default "hybrid_v3_head.pth"
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> int:
    from huggingface_hub import hf_hub_download

    repo = os.environ.get("VERIDEX_HF_REPO", "krishivvv/veridex-deepfake")
    files = [
        os.environ.get("VERIDEX_BACKBONE_FILE", "cnn_baseline_best.pth"),
        os.environ.get("VERIDEX_HEAD_FILE", "hybrid_v3_head.pth"),
    ]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for filename in files:
        dst = MODELS_DIR / filename
        if dst.exists():
            print(f"[prefetch] {filename} already present, skipping.")
            continue
        print(f"[prefetch] downloading {filename} from {repo} ...", flush=True)
        cached = hf_hub_download(repo_id=repo, filename=filename)
        shutil.copy(cached, dst)  # copy out of the HF cache into models/
        print(f"[prefetch] -> {dst} ({dst.stat().st_size // (1024 * 1024)} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
