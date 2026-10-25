# VisionGuard - 22L-6877_A2_Ops

![CI](https://github.com/<you>/22L-6877_A2_Ops/actions/workflows/ci.yml/badge.svg)

A web service that says whether an uploaded image is **AI-generated or real**: a TensorFlow/Keras model
tracked with MLflow on DagsHub, served by Flask with a PostgreSQL prediction trail, packaged with Docker
Compose, and deployed as a cloud model API.

| | |
|---|---|
| GitHub | https://github.com/&lt;you&gt;/22L-6877_A2_Ops |
| DagsHub (code, data, experiments) | https://dagshub.com/&lt;you&gt;/22L-6877_A2_Ops |
| Model API | https://&lt;hf-user&gt;-22l-6877-a2-ops.hf.space |

## Quick start

```bash
docker compose up -d --build
```

Needs `.env` (copy `.env.example`) and `models/champion.keras` (`python -m visionguard.registry export`).
Then open http://127.0.0.1:5000.

## Architecture

```
 browser ──> Flask web app (app/) ──> PostgreSQL  (prediction history)
                  │
                  ├─ INFERENCE_MODE=local : champion.keras loaded once at start-up
                  └─ INFERENCE_MODE=remote: HTTPS + API key ──> model API (api/) on Hugging Face Spaces

 training (src/visionguard) ──> MLflow on DagsHub ──> registry alias "champion" ──> models/champion.keras
 dataset ──> DVC ──> DagsHub storage
```

## Layout

- `src/visionguard/` - audit, EDA, data prep, tf.data pipeline, models, training, tuning, Grad-CAM,
  MLflow helpers, registry, shared preprocessing and inference
- `app/` - Flask app: upload, result, history, health
- `api/` - cloud model API + its Dockerfile
- `labs/` - toy exercises (DVC, Keras Tuner, MLflow, Docker, CI)
- `tests/` - pytest suite (runs without TensorFlow)

Full step-by-step instructions for every task: **[RUNBOOK.md](RUNBOOK.md)**.
