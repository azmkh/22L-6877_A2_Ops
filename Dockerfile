# Task 6.2 - VisionGuard web app image.
# Order matters for the layer cache: the dependency layer (slow, ~600 MB, rarely changes) comes
# before the source code (fast, changes often), so editing a template only rebuilds the last layers.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# 1) dependencies only
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 2) source code (only what serving needs; see .dockerignore)
COPY src/visionguard/__init__.py src/visionguard/preprocess.py src/visionguard/serving.py \
     src/visionguard/validation.py src/visionguard/web.py ./visionguard/
COPY app ./app
COPY wsgi.py .

# No model and no secrets in the image: the model is mounted at /models (compose), and settings
# and passwords arrive as environment variables at run time.
USER appuser
EXPOSE 5000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", \
     "--bind", "0.0.0.0:5000", "--access-logfile", "-", "wsgi:app"]
