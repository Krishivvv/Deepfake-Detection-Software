"""Convenience launcher for the Flask demo.

By default (development) uses the Flask development server. In production
set ``SERVE_WITH_WAITRESS=1`` (already done in the Dockerfile) to serve
through the production-grade ``waitress`` WSGI server instead.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.app import create_app  # noqa: E402
from app.config import Config  # noqa: E402

log = logging.getLogger("app.run")


def main() -> None:
    app = create_app()
    host = Config.HOST
    port = Config.PORT

    if os.environ.get("SERVE_WITH_WAITRESS", "0") == "1":
        from waitress import serve

        log.info("Starting waitress on %s:%s", host, port)
        # Threads tuned for a CPU-heavy single-model server: a couple of
        # parallel uploads but not so many that GIL contention crushes
        # inference latency.
        serve(app, host=host, port=port, threads=4, channel_timeout=300)
    else:
        app.run(host=host, port=port, debug=Config.DEBUG)


if __name__ == "__main__":
    main()
