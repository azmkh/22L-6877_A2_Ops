"""VisionGuard Flask web app (Tasks 4, 5 and 8)."""
import logging

from flask import Flask, render_template
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.config import load_settings
from app.db import Database
from visionguard.web import InMemoryRequest

log = logging.getLogger("visionguard.web")


def build_predictor(cfg):
    """Return (predictor, error). A missing/broken model does not crash the app: it starts,
    logs the reason, and /health reports 503 so the container is marked unhealthy."""
    if cfg["INFERENCE_MODE"] == "remote":
        from app.remote import RemotePredictor

        if not cfg["API_URL"]:
            return None, "INFERENCE_MODE=remote but API_URL is empty"
        log.info("Remote inference via %s", cfg["API_URL"])
        return RemotePredictor(cfg["API_URL"], cfg["API_KEY"], cfg["API_TIMEOUT"]), None
    try:
        from visionguard.serving import LocalPredictor

        predictor = LocalPredictor(cfg["MODEL_PATH"], cfg["MODEL_META_PATH"], cfg["THRESHOLD"])
        log.info("Loaded model %s v%s from %s (threshold %.3f)", predictor.model_name,
                 predictor.model_version, cfg["MODEL_PATH"], predictor.threshold)
        return predictor, None
    except Exception as exc:
        log.error("Could not load model from %s: %s", cfg["MODEL_PATH"], exc)
        return None, f"Model could not be loaded: {exc}"


def create_app(overrides=None, predictor=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config.update(load_settings())
    app.config.update(overrides or {})
    # Whole-request cap (file + multipart overhead); the file itself is checked exactly in validation.
    app.config["MAX_CONTENT_LENGTH"] = int(app.config["MAX_UPLOAD_MB"] * 1024 * 1024) + 64 * 1024

    db = Database(app.config["DATABASE_URL"])
    db.init()
    error = None
    if predictor is None:
        predictor, error = build_predictor(app.config)
    app.extensions["db"] = db
    app.extensions["predictor"] = predictor
    app.extensions["predictor_error"] = error

    from app.routes import bp

    app.register_blueprint(bp)

    @app.context_processor
    def globals_for_templates():
        return {"project_name": app.config["PROJECT_NAME"], "inference_mode": app.config["INFERENCE_MODE"],
                "max_upload_mb": app.config["MAX_UPLOAD_MB"]}

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return render_template("error.html", title="File too large",
                               message=f"Uploads are limited to {app.config['MAX_UPLOAD_MB']:g} MB."), 413

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return render_template("error.html", title=f"{exc.code} {exc.name}", message=exc.description), exc.code

    @app.errorhandler(Exception)
    def unexpected(exc):
        log.exception("Unhandled error")
        return render_template("error.html", title="Something went wrong",
                               message="An unexpected error occurred. It has been logged."), 500

    log.info("VisionGuard started: mode=%s db=%s max_upload=%gMB", app.config["INFERENCE_MODE"],
             db.engine.url.render_as_string(hide_password=True), app.config["MAX_UPLOAD_MB"])
    return app
