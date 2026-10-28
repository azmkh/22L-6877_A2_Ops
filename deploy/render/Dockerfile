# Task 7 - model API image for Render's free plan (512 MB). Build context = repository root:
#   docker build -f deploy/render/Dockerfile -t visionguard-api-lite .
# Same api/app.py as the TensorFlow image (api/Dockerfile), but the champion is the TFLite export
# (scripts/export_tflite.py) run with LiteRT: about 70 MB of memory instead of about 550 MB.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=10000

RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY requirements-api-lite.txt .
RUN pip install --no-cache-dir -r requirements-api-lite.txt

COPY --chown=user src/visionguard/__init__.py src/visionguard/preprocess.py src/visionguard/serving.py \
     src/visionguard/validation.py src/visionguard/web.py ./visionguard/
COPY --chown=user api ./api
COPY --chown=user deploy/render/champion.tflite deploy/render/champion.json ./models/

USER user
ENV MODEL_PATH=/home/user/app/models/champion.tflite
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=4)"

# Shell form so $PORT (set by Render) is expanded. One worker: the model is loaded once.
CMD gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} api.app:app
