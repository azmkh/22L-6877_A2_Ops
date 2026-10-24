# Task 7 - model API image. Build context = repository root (or the folder made by
# deploy/prepare_hf_space.ps1, which has the same layout).
#   docker build -f api/Dockerfile -t visionguard-api .
# The champion model is BUILT IN: cloud hosts such as Hugging Face Spaces have no volume to
# mount, and baking it in makes the deployed image immutable (image tag == model version).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user src/visionguard/__init__.py src/visionguard/preprocess.py src/visionguard/serving.py \
     src/visionguard/validation.py src/visionguard/web.py ./visionguard/
COPY --chown=user api ./api
COPY --chown=user models/champion.keras models/champion.json ./models/

USER user
ENV MODEL_PATH=/home/user/app/models/champion.keras
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=4)"

# Shell form so $PORT (set by Render, 7860 on Hugging Face) is expanded.
CMD gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} api.app:app
