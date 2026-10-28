# Assignment 2 progress audit

Audited against `Assignment 2.pdf` on 2026-10-25. This is a repository evidence audit; it does not
claim screenshots, cloud activity, or report content that is not present in the checkout.

## Repository state

- Git branch: `main`, at `3e8ec7c` (`Add DagsHub DVC remote; write UTF-8 output so remote MLflow runs finish`).
- The working tree was clean before this audit. `reports/assignment_audit.md` is the audit deliverable.
- The assignment PDF asks for a separately submitted, screenshot-supported `22L-6877_VisionGuard.pdf`.
  No such report is present in this repository.

## Task-by-task status

| Assignment | Repository evidence | Status / remaining work |
|---|---|---|
| 1.1 Environment | `.venv`, `requirements*.txt`, `labs/env_check.py` | Script and environment exist; capture exact version/GPU output for the report. |
| 1.2 Data audit | `src/visionguard/data_audit.py`, `reports/audit/*` | Implemented and report artifacts exist: 975 files, 0 corrupt, 1 duplicate copy, no cross-class duplicates. |
| 1.3 EDA | `src/visionguard/eda.py`, `reports/eda/*.png` | Figures exist; add captions and the requested short interpretations to the final report. |
| 1.4 DVC | `data/raw.dvc`, `.dvc`, `labs/dvc_toy.ps1` | Dataset pointer and DVC setup exist. Evidence of toy workflow, remote push/pull, and report explanation must be supplied. |
| 2.1 Pipeline | `src/visionguard/pipeline.py`, `reports/pipeline/augmented_batch.png` | Implemented; explain pipeline, augmentation, input size, and aspect-ratio handling in report. |
| 2.2-2.6 Models | `src/visionguard/{models,train,evaluate}.py`; runs under `outputs/` | Baseline and MobileNetV2 training/evaluation artifacts exist, including per-epoch history and test analyses. Explain design choices and include the required results/figures in report. |
| 2.7 Keras Tuner | `src/visionguard/tune.py`, `labs/keras_tuner_toy.py`, `reports/tuner/*` | Tuner code and result tables exist; document trial count, best settings, and validation overfit risk. |
| Bonus Grad-CAM | `src/visionguard/gradcam.py`, `reports/gradcam.png` | Artifact exists; add interpretation to report. |
| 3.1-3.2 MLflow/DagsHub | `labs/mlflow_toy.py`, `src/visionguard/tracking.py`, `.env.example` | Code/configuration exists. The live toy runs and DagsHub screenshots cannot be verified from repository files. |
| 3.3-3.4 Tracking/artifacts | Training logger, autolog option, model/split/environment artifact code | Implemented in code. Confirm the two tracking modes and required artifacts are visible in DagsHub. |
| 3.5 Experiment comparison | `scripts/run_experiments.ps1`, `src/visionguard/compare_runs.py`, `reports/comparison/{runs.csv,runs.md,runs.png}` | **Six distinct conditions completed and compared:** baseline, reference MobileNetV2, lower learning rate, no augmentation, frozen backbone, and 160-pixel input. The comparison selects `exp6_mnv2_160px` by validation ROC-AUC (0.8811); its test ROC-AUC is 0.8509 and test F1 is 0.7429. At the validation-selected threshold, test FPR is 10.6%, slightly above the 10% validation cap; report this held-out result as observed, not as a guarantee. `compare_runs.py` now collapses repeated identical configurations to the newest run so reruns do not inflate the experiment count. |
| 3.6 Registry | `src/visionguard/registry.py`, `scripts/predict_champion.py` | Workflow code exists. Verify at least two registry versions, `champion` alias, and successful load/prediction in DagsHub. |
| 4-5 Flask and database | `app/`, `src/visionguard/{web,serving}.py`, route/database tests | Application and persistence implementation exist. Capture browser evidence for success/failure cases, health, and history. |
| 6 Docker/Compose/CI | `Dockerfile`, `docker-compose.yml`, `Dockerfile.naive`, `.github/workflows/ci.yml`, `labs/docker_warmup.md` | Project configuration exists. Build/runtime checks, image-size comparison, operational troubleshooting, and screenshots remain report evidence items. |
| 7-8 Cloud API and integration | `api/`, `deploy/prepare_hf_space.ps1`, `postman/VisionGuard_API.postman_collection.json`, remote inference support | Deployment and integration code exists. A live URL, public/other-network API check, Postman evidence, and remote-mode history screenshots are not verifiable from the checkout. |
| 11 Report | No final report PDF in repo | **Incomplete deliverable:** prepare the named PDF with front matter, one chapter per task, explanations, captions, screenshots, links, graded commit, and AI-assistance disclosure. |

## Findings from the 2026-10-27 re-check

- **Live API is paused.** The Space `azmkh7/22L-6877_A2_Ops` (Gradio SDK) exists, but
  `https://azmkh7-22l-6877-a2-ops.hf.space/health` answers "The space is paused". It must be
  restarted from the Space settings before the Postman, other-network and Task 8 evidence can be
  captured. Its published files also include `index.html` and `style.css`, which are not in
  `deploy/hf_gradio_upload/`.
- **Comparison regenerated.** The earlier table (17:11) predated `exp7`, `exp8` and the second
  `exp9` run. `reports/comparison/` now holds all nine conditions. `exp6_mnv2_160px` still has the
  best validation ROC-AUC (0.8811); `exp9_mnv2_tuned` (run `b9d0a86a…`) is close on ROC-AUC (0.8722)
  and best on validation F1 (0.7832).
- **Registry moved to DagsHub (resolved 2026-10-27).** Versions 1-3 had been registered only in the
  local `mlruns` store; the DagsHub registry was empty, and the old champion (`eb6e9c52…`) was weaker
  on validation than `exp6` and the later `exp9`. DagsHub now has `22L-6877_A2_Ops_visionguard`
  version 1 = `exp6_mnv2_160px` (`be0d107a…`) and version 2 = `exp9_mnv2_tuned` (`b9d0a86a…`) with
  alias `champion`. `registry export` wrote it to `models/` (the previous files are kept in
  `models/previous_v3_local/`); `predict_champion.py`, `visionguard.serving` and the Flask routes
  (health, two predictions, 415/422/413 rejections, history) were checked against it.
- **Space has champion v2 but cannot run.** The model files were uploaded (Space commit `42ca021`),
  but Hugging Face refuses to start the Space: the free CPU Basic quota for `azmkh7` is 0, and
  ZeroGPU needs an account at least 30 days old (or PRO / a community grant). Until then there is no
  public API URL for the Task 7 and 8 evidence.
- **Local integration test (2026-10-28).** The Space's `app.py` was run locally with the Space's
  requirements and model v2. `/health` reported v2, threshold 0.701. `/predict` classified a real and
  an AI image correctly and returned 401 (no or wrong key), 415 (.txt) and 422 (undecodable image);
  Bearer auth worked. Flask with `INFERENCE_MODE=remote` against it classified 2 real and 2 AI test
  images correctly, logged all 4 History rows as `remote:127.0.0.1:7861`, rejected a .txt with 415,
  and showed a 502 error page when the API was unreachable.
- Keras Tuner toy run completed: best 128 units, lr 3.07e-4, val accuracy 0.834, test 0.836.
- Docker is not installed on this machine, so the Task 6 build, size and Compose evidence must be
  captured on a machine with Docker Desktop.
- **CI toy done (2026-10-28).** `labs/ci_toy/` was pushed to the public repository
  https://github.com/azmkh/22L-6877_A2_Ops_ci_toy. Three pushes gave three runs: `7b50db3` passed on
  Python 3.11 and 3.12; `43e5818` (`add()` changed to subtract) failed at `pytest -v` on 3.11, and the
  3.12 job was cancelled by the matrix's fail-fast default; `9f9ed14` (fix) passed on both.
- `.venv` reports Python 3.12.10, TensorFlow 2.19.0, Keras 3.10.0, no GPU.
- The DVC toy used a local folder remote (`~/dvc_toy_A2_remote`), not DagsHub; push and pull worked.

## Next actions

1. Verify the registry, app, Compose stack, and public API in their real connected environments and capture the assignment's evidence screenshots.
2. Assemble and visually review `22L-6877_VisionGuard.pdf`; include only claims supported by the captured evidence.

## Blocker from the attempted continuation

The first `exp4_mnv2_noaug` attempt used the configured remote tracking URI from the ignored `.env`.
MLflow could not resolve `dagshub.com` and stopped before model fitting. After approved network access
was available, the run completed and logged successfully. The earlier partial folder
`outputs/20261025-151853_exp4_mnv2_noaug/` contains only a model summary and layer table; the complete
run is `outputs/20261025-152440_exp4_mnv2_noaug/` (DagsHub run ID
`bdd928d082394380927491e2863f8dc0`). No credentials were printed.

The additional completed run is `outputs/20261025-165057_exp6_mnv2_160px/` (DagsHub run ID
`be0d107a9ef04ebca4e2335fa4361ee9`). The corrected comparison was logged as run
`f75ab29597b94cdca9b5d2d26de412a0`.
