"""Task 7 - VisionGuard model API, deployed to the cloud (Hugging Face Spaces / Render).

    GET  /health                 model info, no key needed
    POST /predict                multipart field "image" (or the raw image as the body)
                                 header  X-API-Key: <API_KEY>   (or Authorization: Bearer <API_KEY>)

The model file is built into the image and loaded once at start-up.
"""
import hmac
import logging
import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from visionguard.serving import LocalPredictor
from visionguard.validation import ALLOWED_EXTENSIONS, InvalidImage, check_image_bytes, extension_of
from visionguard.web import InMemoryRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [api] %(message)s")
log = logging.getLogger("visionguard.api")


def error(status, message):
    return jsonify({"error": message, "status": status}), status


def create_app(predictor=None):
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    max_bytes = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
    app.config["MAX_CONTENT_LENGTH"] = max_bytes + 64 * 1024
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        log.warning("API_KEY is not set - every /predict call will be refused")
    if predictor is None:
        threshold = os.getenv("THRESHOLD", "").strip()
        predictor = LocalPredictor(os.getenv("MODEL_PATH", "models/champion.keras"),
                                   os.getenv("MODEL_META_PATH") or None,
                                   float(threshold) if threshold else None)
    log.info("Model %s v%s ready", predictor.model_name, predictor.model_version)

    def authorized():
        supplied = request.headers.get("X-API-Key", "")
        auth = request.headers.get("Authorization", "")
        if not supplied and auth.lower().startswith("bearer "):
            supplied = auth[7:]
        # constant-time comparison so the key can't be guessed from response timing
        return bool(api_key) and hmac.compare_digest(supplied.encode(), api_key.encode())

    @app.get("/")
    def index():
        return jsonify({"service": "VisionGuard model API", "health": "/health",
                        "predict": "POST /predict (multipart 'image', header X-API-Key)"})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", **predictor.describe()})

    @app.post("/predict")
    def predict():
        if not authorized():
            log.warning("Refused /predict from %s: bad or missing API key", request.remote_addr)
            return error(401, "Missing or invalid API key.")
        upload = request.files.get("image")
        if upload is not None:
            filename = secure_filename(upload.filename or "")
            if extension_of(filename) not in ALLOWED_EXTENSIONS:
                return error(415, f"File type not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
            data = upload.read(max_bytes + 1)
        else:
            filename, data = "", request.get_data()
        if not data:
            return error(400, "No image received. Send multipart field 'image'.")
        if len(data) > max_bytes:
            return error(413, f"Image larger than {max_bytes // (1024 * 1024)} MB.")
        try:
            check_image_bytes(data)
        except InvalidImage as exc:
            return error(422, str(exc))
        result = predictor.predict(data, filename)
        log.info("predict %s -> %s %.3f (%s ms)", filename or "<raw>", result["label"],
                 result["confidence"], result["latency_ms"])
        return jsonify(result)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return error(exc.code, exc.description)

    return app


if os.getenv("VISIONGUARD_API_NO_AUTOLOAD") != "1":
    app = create_app()
