"""Task 4.2 / 5 / 8 - pages: upload, result, history, health."""
import hashlib
import logging
from datetime import datetime, timezone

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, url_for

from app.validation import UploadError, validate_upload

bp = Blueprint("main", __name__)
log = logging.getLogger("visionguard.web")


def _db():
    return current_app.extensions["db"]


@bp.get("/")
def index():
    return render_template("index.html")


@bp.post("/predict")
def predict():
    max_bytes = int(current_app.config["MAX_UPLOAD_MB"] * 1024 * 1024)
    try:
        data, filename = validate_upload(request.files.get("image"), max_bytes)
    except UploadError as exc:
        log.info("Rejected upload: %s (%s)", exc.title, exc.message)
        return render_template("error.html", title=exc.title, message=exc.message), exc.status

    predictor = current_app.extensions["predictor"]
    if predictor is None:
        return render_template("error.html", title="Model unavailable",
                               message=current_app.extensions["predictor_error"]), 503
    try:
        result = predictor.predict(data, filename)
    except Exception as exc:
        log.error("Inference failed for %s: %s", filename, exc)
        return render_template("error.html", title="Inference failed", message=str(exc)), 502

    row = _db().add(
        filename=filename,
        file_sha256=hashlib.sha256(data).hexdigest(),
        file_size=len(data),
        label=result["label"],
        probability_ai=result["probability_ai"],
        confidence=result["confidence"],
        threshold=result["threshold"],
        model_name=str(result["model_name"]),
        model_version=str(result["model_version"]),
        inference_source=result["source"],
        latency_ms=result["latency_ms"],
    )
    log.info("Prediction #%d %s -> %s (%.1f%%) by %s v%s via %s", row.id, filename, result["label"],
             result["confidence"] * 100, row.model_name, row.model_version, row.inference_source)
    # Post/Redirect/Get: refreshing the result page doesn't resubmit (and re-log) the upload.
    return redirect(url_for("main.result", prediction_id=row.id))


@bp.get("/result/<int:prediction_id>")
def result(prediction_id):
    row = _db().get(prediction_id)
    if row is None:
        abort(404, description="No prediction with that id.")
    return render_template("result.html", p=row)


@bp.get("/history")
def history():
    label = request.args.get("label")
    if label not in ("ai", "real"):
        label = None
    db = _db()
    return render_template("history.html", rows=db.recent(200, label), counts=db.counts(), label=label)


@bp.get("/health")
def health():
    db_ok = _db().ping()
    predictor = current_app.extensions["predictor"]
    healthy = db_ok and predictor is not None
    body = {
        "status": "ok" if healthy else "degraded",
        "database": "ok" if db_ok else "error",
        "model_loaded": predictor is not None,
        "inference_mode": current_app.config["INFERENCE_MODE"],
        "model": predictor.describe() if predictor else None,
        "error": current_app.extensions["predictor_error"],
        "debug": current_app.debug,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return jsonify(body), 200 if healthy else 503
