"""Convenience launcher for the Flask demo: ``python run_app.py``."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.app import create_app  # noqa: E402
from app.config import Config  # noqa: E402


def main() -> None:
    app = create_app()
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)


if __name__ == "__main__":
    main()
