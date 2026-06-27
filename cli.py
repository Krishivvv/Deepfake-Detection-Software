"""
Veridex command-line entry point — a thin dispatcher over the task scripts.

    python cli.py train            --model cnn [trainer args...]
    python cli.py evaluate         --model hybrid_v3 [eval args...]
    python cli.py extract-features --backbone cnn [args...]
    python cli.py serve            [flask args...]

It simply forwards the remaining arguments to the underlying script using the
current Python interpreter, so there is one documented surface for the whole
pipeline while each script stays independently runnable.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# subcommand -> (script, fixed leading args)
COMMANDS: dict[str, tuple[str, list[str]]] = {
    "train-cnn": ("train_cnn.py", []),
    "train-hybrid": ("train_hybrid_v2.py", []),
    "evaluate": ("evaluate.py", []),
    "extract-features": ("extract_features.py", []),
    "tune-threshold": ("tune_cnn_threshold.py", []),
    "serve": ("run_app.py", []),
    "verify": ("verify_environment.py", []),
}


def _usage() -> str:
    lines = ["Veridex CLI", "Usage: python cli.py <command> [args...]", "", "Commands:"]
    for name in COMMANDS:
        lines.append(f"  {name}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  train-cnn      forwards to train_cnn.py")
    lines.append("  train-hybrid   forwards to train_hybrid_v2.py (cached-feature LSTM head)")
    lines.append("  evaluate       --model {cnn,hybrid,hybrid_v3}")
    lines.append("  extract-features --backbone {imagenet,cnn}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    script, fixed = COMMANDS[cmd]
    target = PROJECT_ROOT / script
    if not target.exists():
        print(f"Script not found: {target}", file=sys.stderr)
        return 1

    return subprocess.call([sys.executable, str(target), *fixed, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
