"""
One-shot deploy of the Veridex demo to Hugging Face.

Run this ONCE from a machine that (a) is logged in (`huggingface-cli login`)
and (b) has the trained weights under ``models/`` — i.e. this machine. It:

  1. creates the model repo ``krishivvv/veridex-deepfake`` (if missing) and
     uploads the two weight files (skipping any already present),
  2. creates the Space ``krishivvv/Veridex`` (Gradio SDK) if missing and pushes
     ONLY the code the demo needs (no weights, no dataset),
  3. prints the variables to set in the Space settings.

Idempotent: re-running re-uploads only changed files. The token is read from
the local HF login / ``HF_TOKEN`` env var — it is never printed or committed.

    python scripts/deploy_space.py            # do it
    python scripts/deploy_space.py --dry-run  # show the plan only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_REPO = os.environ.get("VERIDEX_HF_REPO", "krishivvv/veridex-deepfake")
SPACE_REPO = os.environ.get("VERIDEX_HF_SPACE", "krishivvv/Veridex")

WEIGHTS = ["cnn_baseline_best.pth", "hybrid_v3_head.pth"]

# Code paths uploaded to the Space (file or directory, relative to repo root).
SPACE_FILES = [
    "gradio_app.py",
    "config.yaml",
    "requirements-space.txt",
    "README_HF_SPACE.md",
    "src",
    "app/__init__.py",
    "app/config.py",
    "app/utils",
]


def _check_weights() -> list[Path]:
    missing = [w for w in WEIGHTS if not (PROJECT_ROOT / "models" / w).exists()]
    if missing:
        sys.exit(f"ERROR: missing weights under models/: {missing}")
    return [PROJECT_ROOT / "models" / w for w in WEIGHTS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--code-only", action="store_true",
                    help="Only sync Space code (skip model repo + weights). "
                         "Used by CI, where local weights are unavailable.")
    args = ap.parse_args()

    weight_paths = [] if args.code_only else _check_weights()
    print(f"Model repo : {'(skipped)' if args.code_only else MODEL_REPO}")
    print(f"Space      : {SPACE_REPO}")
    print(f"Weights    : {[p.name for p in weight_paths] or '(skipped)'}")
    print(f"Space files: {SPACE_FILES}")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        sys.exit("ERROR: not logged in. Run `huggingface-cli login` first.")
    api = HfApi(token=token)

    # 1. Model repo + weights (skipped in --code-only mode).
    if not args.code_only:
        api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
        for p in weight_paths:
            print(f"Uploading {p.name} -> {MODEL_REPO} ...")
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                            repo_id=MODEL_REPO, repo_type="model")

    # 2. Space + code.
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
    for rel in SPACE_FILES:
        src = PROJECT_ROOT / rel
        if not src.exists():
            print(f"  skip (missing): {rel}")
            continue
        if src.is_dir():
            print(f"Uploading folder {rel}/ -> {SPACE_REPO} ...")
            api.upload_folder(folder_path=str(src), path_in_repo=rel,
                              repo_id=SPACE_REPO, repo_type="space",
                              ignore_patterns=["__pycache__", "*.pyc"])
        else:
            # HF Spaces expect README.md (with YAML front-matter) and install
            # from requirements.txt — remap our source filenames accordingly.
            dest = {
                "README_HF_SPACE.md": "README.md",
                "requirements-space.txt": "requirements.txt",
            }.get(rel, rel)
            print(f"Uploading {rel} -> {SPACE_REPO}:{dest} ...")
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                            repo_id=SPACE_REPO, repo_type="space")

    print("\nDone. Set these in the Space Settings -> Variables (or via the API):")
    print(f"  VERIDEX_HF_REPO={MODEL_REPO}")
    print("  APP_MODEL_KIND=hybrid_v3")
    print(f"\nSpace: https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
