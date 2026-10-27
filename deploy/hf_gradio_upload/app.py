"""Gradio ZeroGPU wrapper that retains VisionGuard's authenticated REST API."""
import hmac
import os
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from serving import LocalPredictor
from validation import ALLOWED_EXTENSIONS, InvalidImage, check_image_bytes, extension_of


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "champion.keras")))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", str(BASE_DIR / "champion.json")))
MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
API_KEY = os.getenv("API_KEY", "")
predictor = LocalPredictor(MODEL_PATH, MODEL_META_PATH)

api = FastAPI(title="VisionGuard API", version="1.0.0")


def error(status: int, message: str):
    return JSONResponse({"error": message, "status": status}, status_code=status)


def is_authorized(request: Request) -> bool:
    supplied = request.headers.get("X-API-Key", "")
    auth = request.headers.get("Authorization", "")
    if not supplied and auth.lower().startswith("bearer "):
        supplied = auth[7:]
    return bool(API_KEY) and hmac.compare_digest(supplied.encode(), API_KEY.encode())


@api.get("/health")
async def health():
    return {"status": "ok", **predictor.describe(), "source": "huggingface-space"}


@api.post("/predict")
async def predict(request: Request):
    if not is_authorized(request):
        return error(401, "Missing or invalid API key.")

    filename = ""
    content_type = request.headers.get("content-type", "")
    if content_type.lower().startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if not isinstance(upload, UploadFile):
            return error(400, "Send the image as multipart field 'image'.")
        filename = Path(upload.filename or "").name
        if extension_of(filename) not in ALLOWED_EXTENSIONS:
            return error(415, f"File type not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
        data = await upload.read(MAX_BYTES + 1)
    else:
        data = await request.body()

    if not data:
        return error(400, "No image received. Send multipart field 'image'.")
    if len(data) > MAX_BYTES:
        return error(413, f"Image larger than {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        check_image_bytes(data)
    except InvalidImage as exc:
        return error(422, str(exc))

    result = predictor.predict(data, filename or None)
    result["source"] = "huggingface-space"
    return result


def classify(image_path: str | None):
    if not image_path:
        return {}
    data = Path(image_path).read_bytes()
    check_image_bytes(data)
    result = predictor.predict(data, Path(image_path).name)
    return {result["label_display"]: result["confidence"]}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="filepath", label="Choose an image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="VisionGuard: AI-generated or real?",
    description="Upload an image to classify it. The authenticated REST API is available at /predict.",
)

app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
