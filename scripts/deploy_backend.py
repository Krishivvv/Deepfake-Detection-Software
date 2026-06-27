"""
Deploy the Veridex Flask + model API to a Hugging Face Docker Space.

Run once from a machine logged in to HF (`huggingface-cli login`). Creates the
Docker Space ``krishivvv/veridex-api`` (if missing) and uploads only the
backend code + Dockerfile. Weights are NOT uploaded — the container pulls them
from the public model repo ``krishivvv/veridex-deepfake`` at startup.

    python scripts/deploy_backend.py            # deploy
    python scripts/deploy_backend.py --dry-run  # show plan only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPACE_REPO = os.environ.get("VERIDEX_API_SPACE", "krishivvv/veridex-api")

# file/dir -> path in the Space repo (README_HF_BACKEND.md becomes README.md)
SPACE_FILES = {
    "Dockerfile": "Dockerfile",
    "requirements.txt": "requirements.txt",
    "README_HF_BACKEND.md": "README.md",
    ".dockerignore": ".dockerignore",
    "run_app.py": "run_app.py",
    "app": "app",
    "src": "src",
    "scripts/prefetch_weights.py": "scripts/prefetch_weights.py",
}

# Never ship these into the Space (covered by .dockerignore too, but be safe).
IGNORE = ["__pycache__", "*.pyc", "users.db", "logs/*", "static/uploads/*"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Space      : {SPACE_REPO} (docker sdk)")
    print(f"Files      : {list(SPACE_FILES)}")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        sys.exit("ERROR: not logged in. Run `huggingface-cli login` first.")
    api = HfApi(token=token)

    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)
    for src_rel, dest in SPACE_FILES.items():
        src = PROJECT_ROOT / src_rel
        if not src.exists():
            print(f"  skip (missing): {src_rel}")
            continue
        if src.is_dir():
            print(f"Uploading folder {src_rel}/ ...")
            api.upload_folder(folder_path=str(src), path_in_repo=dest,
                              repo_id=SPACE_REPO, repo_type="space",
                              ignore_patterns=IGNORE)
        else:
            print(f"Uploading {src_rel} -> {dest} ...")
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                            repo_id=SPACE_REPO, repo_type="space")

    print("\nDone. Set these in the Space Settings -> Variables/Secrets:")
    print("  APP_CORS_ORIGINS=<your Vercel URL>   (or * for any origin)")
    print("  APP_SECRET_KEY=<long random string>  (secret)")
    print(f"\nSpace: https://huggingface.co/spaces/{SPACE_REPO}")
    print(f"API:   https://{SPACE_REPO.replace('/', '-')}.hf.space/health")


if __name__ == "__main__":
    main()
