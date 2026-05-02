"""
Flask web demo for deepfake detection.

Endpoints:
    GET  /              upload page
    POST /predict       JSON: predict on uploaded video
    GET  /about         model info page
    GET  /health        liveness probe
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
import uuid
from pathlib import Path

from flask import (
    Flask, jsonify, render_template, request, send_from_directory, url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config, load_threshold, predictor_kwargs  # noqa: E402
from app.utils.predictor import (  # noqa: E402
    ModelLoadError, PredictionError, build_predictor,
)
from app.utils.preprocessor import (  # noqa: E402
    PreprocessingError, VideoPreprocessor,
)


def _setup_logging(log_dir: Path) -> tuple[logging.Logger, logging.Logger]:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    err = logging.handlers.RotatingFileHandler(
        log_dir / "error.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8",
    )
    err.setLevel(logging.ERROR); err.setFormatter(fmt)
    pred = logging.handlers.RotatingFileHandler(
        log_dir / "predictions.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8",
    )
    pred.setLevel(logging.INFO); pred.setFormatter(fmt)

    app_log = logging.getLogger("app")
    app_log.setLevel(logging.INFO); app_log.addHandler(err)

    pred_log = logging.getLogger("app.predictions")
    pred_log.setLevel(logging.INFO); pred_log.addHandler(pred); pred_log.propagate = False

    console = logging.StreamHandler()
    console.setLevel(logging.INFO); console.setFormatter(fmt)
    app_log.addHandler(console)
    pred_log.addHandler(console)

    return app_log, pred_log


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def create_app() -> Flask:
    Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    app_log, pred_log = _setup_logging(Config.LOG_DIR)

    app = Flask(
        __name__,
        template_folder=str(APP_ROOT / "templates"),
        static_folder=str(APP_ROOT / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.jinja_env.globals["app_threshold"] = lambda: app.config.get("MODEL_THRESHOLD", 0.5)

    kind = Config.MODEL_KIND
    threshold = load_threshold(kind=kind)
    app.config["MODEL_KIND"] = kind
    app.config["MODEL_THRESHOLD"] = threshold
    app_log.info("Model kind: %s | decision threshold: %.3f", kind, threshold)

    try:
        predictor = build_predictor(kind, **predictor_kwargs(kind))
        preprocessor = VideoPreprocessor(
            num_frames=Config.NUM_FRAMES,
            image_size=Config.IMAGE_SIZE,
            device="cpu",
        )
        app.config["PREDICTOR"] = predictor
        app.config["PREPROCESSOR"] = preprocessor
        app_log.info("Model and preprocessor initialised.")
    except ModelLoadError as exc:
        app_log.exception("Model failed to load at startup")
        app.config["PREDICTOR"] = None
        app.config["PREPROCESSOR"] = None
        app.config["STARTUP_ERROR"] = str(exc)

    @app.route("/")
    def index():
        startup_error = app.config.get("STARTUP_ERROR")
        return render_template(
            "index.html",
            startup_error=startup_error,
            max_mb=Config.MAX_CONTENT_LENGTH // (1024 * 1024),
            allowed_exts=sorted(Config.ALLOWED_EXTENSIONS),
            threshold=app.config["MODEL_THRESHOLD"],
        )

    @app.route("/about")
    def about():
        predictor = app.config.get("PREDICTOR")
        meta = getattr(predictor, "config_meta", {}) if predictor else {}
        notes = getattr(predictor, "notes", "") if predictor else ""
        return render_template(
            "about.html",
            meta=meta, notes=notes,
            kind=app.config["MODEL_KIND"],
            threshold=app.config["MODEL_THRESHOLD"],
            num_frames=Config.NUM_FRAMES,
            image_size=Config.IMAGE_SIZE,
        )

    @app.route("/health")
    def health():
        ok = app.config.get("PREDICTOR") is not None
        return jsonify({"ok": ok, "model_loaded": ok}), 200 if ok else 503

    @app.route("/predict", methods=["POST"])
    def predict():
        request_id = uuid.uuid4().hex[:8]
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?")
        t0 = time.time()

        predictor: HybridPredictor | None = app.config.get("PREDICTOR")
        preprocessor: VideoPreprocessor | None = app.config.get("PREPROCESSOR")
        if predictor is None or preprocessor is None:
            return _err(503, "model_not_loaded",
                        app.config.get("STARTUP_ERROR",
                                       "Model is not initialised. Server is starting up."))

        if "video" not in request.files:
            return _err(400, "missing_file",
                        "No 'video' field in form upload.")

        f = request.files["video"]
        if not f or f.filename == "":
            return _err(400, "empty_filename", "No file was selected.")

        if not _allowed_file(f.filename):
            return _err(400, "invalid_type",
                        "Please upload a video file ("
                        + ", ".join(f".{e}" for e in sorted(Config.ALLOWED_EXTENSIONS)) + ").")

        # Probe first byte to fail fast on empty uploads.
        f.stream.seek(0, os.SEEK_END)
        size = f.stream.tell()
        f.stream.seek(0)
        if size == 0:
            return _err(400, "empty_file", "Uploaded file is empty.")
        if size > Config.MAX_CONTENT_LENGTH:
            return _err(413, "file_too_large",
                        f"Video must be under {Config.MAX_CONTENT_LENGTH // (1024*1024)} MB.")

        ext = f.filename.rsplit(".", 1)[1].lower()
        safe_name = secure_filename(f"{request_id}.{ext}")
        save_path = Config.UPLOAD_DIR / safe_name
        try:
            f.save(str(save_path))
        except Exception as exc:  # noqa: BLE001
            app_log.exception("[%s] failed to save upload", request_id)
            return _err(500, "save_failed",
                        f"Could not save the upload: {exc!s}")

        try:
            try:
                clip, info = preprocessor.preprocess(save_path)
            except PreprocessingError as pe:
                pred_log.info(
                    "[%s] PREPROCESS_ERROR file=%s ip=%s code=%s msg=%s",
                    request_id, safe_name, client_ip, pe.code, pe.message,
                )
                return _err(400, pe.code, pe.message)

            try:
                result = predictor.predict(clip, threshold=app.config["MODEL_THRESHOLD"])
            except PredictionError as pe:
                pred_log.exception("[%s] PREDICTION_ERROR", request_id)
                return _err(500, "inference_failed", str(pe))
            except Exception:  # noqa: BLE001
                pred_log.exception("[%s] PREDICTION_ERROR_UNHANDLED", request_id)
                return _err(500, "inference_failed",
                            "Prediction failed. Please try a different video.")

            elapsed = time.time() - t0
            payload = {
                "ok": True,
                "request_id": request_id,
                "label": result["label"],
                "confidence_pct": round(result["confidence"] * 100, 2),
                "probability_fake_pct": round(result["probability_fake"] * 100, 2),
                "probability_real_pct": round(result["probability_real"] * 100, 2),
                "threshold": result["threshold"],
                "device": result["device"],
                "inference_seconds": round(result["inference_seconds"], 3),
                "total_seconds": round(elapsed, 3),
                "video_info": info,
            }
            pred_log.info(
                "[%s] OK file=%s ip=%s label=%s conf=%.2f%% p_fake=%.4f "
                "frames_with_faces=%d total=%.2fs",
                request_id, safe_name, client_ip,
                payload["label"], payload["confidence_pct"],
                result["probability_fake"], info["frames_with_faces"], elapsed,
            )
            return jsonify(payload), 200

        finally:
            _safe_unlink(save_path)

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_e):
        return _err(413, "file_too_large",
                    f"Video must be under {Config.MAX_CONTENT_LENGTH // (1024*1024)} MB.")

    @app.errorhandler(404)
    def _not_found(_e):
        return _err(404, "not_found", "The requested page was not found.")

    @app.errorhandler(500)
    def _server(_e):
        app_log.exception("Unhandled server error")
        return _err(500, "internal_error",
                    "Internal server error. Please try again.")

    return app


def _err(http_status: int, code: str, message: str):
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if request.accept_mimetypes.accept_html and not request.is_json and \
       request.path != "/predict":
        # Direct browse to a missing page → render error template.
        return render_template("error.html", code=code, message=message), http_status
    return jsonify(payload), http_status


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
