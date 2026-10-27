---
title: 22L-6877 A2 Ops
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: 3.12
suggested_hardware: zero-a10g
app_file: app.py
---

# VisionGuard

Public image classifier and model API for the 22L-6877 MLOps assignment.

The Gradio page provides an interactive demo. The API remains compatible with the
project's Flask client:

- `GET /health` returns model metadata.
- `POST /predict` accepts a multipart image field named `image` (or raw image bytes).
- Supply the Space secret `API_KEY` in `X-API-Key` or `Authorization: Bearer ...`.

The Space runs on Hugging Face ZeroGPU. The TensorFlow model executes on CPU; ZeroGPU
is used as the free Gradio Space runtime tier.
