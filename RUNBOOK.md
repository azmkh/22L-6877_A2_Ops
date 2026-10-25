# VisionGuard runbook - step by step, with the evidence to capture

Commands are for **PowerShell on Windows**, run from the repository root. Every screenshot must show
your username / terminal prompt, the date or time, and the project name (assignment 11.1).

> Roll number **22L-6877**: the project name is `22L-6877_A2_Ops` (`PROJECT_NAME` in `.env`). Use that
> same name for the GitHub repo, DagsHub repo, MLflow experiment, Hugging Face Space and report.

| Where the code is | What it does |
|---|---|
| `src/visionguard/` | training package: audit, EDA, data prep, pipeline, models, train, tune, Grad-CAM, registry |
| `app/` | Flask web app (upload, result, history, health) + database trail |
| `api/` | cloud model API (Task 7) |
| `labs/` | toy exercises: env check, DVC, Keras Tuner, MLflow, Docker warm-up, CI toy |
| `scripts/` | experiment grid, load-by-alias script |
| `deploy/` | Hugging Face Space packaging |
| `tests/`, `.github/workflows/ci.yml` | tests + CI |

---

## 0. One-time setup

1. Install **Python 3.11** (TensorFlow 2.19 does not support 3.14, which is what this PC has now):
   `winget install Python.Python.3.11`, or install [uv](https://docs.astral.sh/uv/) and let it fetch Python.
2. Install **Git**, **Docker Desktop** (with WSL2), **Postman**, and create accounts on GitHub, DagsHub,
   Kaggle and Hugging Face.
3. Create the GitHub repo `22L-6877_A2_Ops` and push this folder:
   ```powershell
   git init -b main
   git add . ; git commit -m "VisionGuard skeleton"
   git remote add origin https://github.com/<you>/22L-6877_A2_Ops.git
   git push -u origin main
   ```
4. On DagsHub: **Create → New repository → Connect a repository → GitHub** and pick `22L-6877_A2_Ops`.
   This gives you MLflow tracking, DVC storage and the code view in one place (Tasks 1.4 and 3.2).
5. `Copy-Item .env.example .env` and fill it in. `.env` is git-ignored and docker-ignored.

## Task 1 - Data understanding and versioning

### 1.1 Environment
```powershell
py -3.11 -m venv .venv            # or: uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                  # makes `visionguard` importable everywhere
python labs\env_check.py
```
**Evidence:** screenshot of `env_check.py` (versions + GPU line). On native Windows TensorFlow
reports no GPU (no GPU builds since 2.11) - say so; CPU is enough for this dataset size.

### 1.2 Dataset audit
Download the [Kaggle dataset](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)
zip and extract it so the class folders sit directly in `data\raw\` (e.g. `data\raw\AiArtData\...`,
`data\raw\RealArt\...`). Any folder name containing "real" / "ai" is mapped automatically.
```powershell
python -m visionguard.data_audit
```
**Evidence:** `reports\audit\summary.md` (counts per class; min/median/max of width, height, file size;
corrupt and duplicate counts). Full table: `reports\audit\audit.csv`; problem files in `corrupt.csv`,
`duplicates.csv`.

### 1.3 Visual EDA
```powershell
python -m visionguard.eda
```
**Evidence:** `reports\eda\samples_per_class.png`, `histograms.png`, `formats_modes.png` - each with a
caption and 2-3 sentences of interpretation (e.g. do the classes differ in size or format? That is a
shortcut the model could learn instead of real "AI-ness").

### 1.4 DVC
**Toy part** (new empty folder, local-folder remote): `.\labs\dvc_toy.ps1` and screenshot each step.
To also use DagsHub for the toy, create an empty DagsHub repo `22L-6877_A2_Ops_dvc_toy`, push the toy
git repo to it (`git remote add origin https://dagshub.com/<you>/22L-6877_A2_Ops_dvc_toy.git; git push -u origin main`),
and add its storage as a second remote (same commands as below).

**Apply to the dataset** (in this repo):
```powershell
dvc init
dvc add data\raw
git add data\raw.dvc data\.gitignore .dvc .dvcignore
git commit -m "Track dataset with DVC"
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<you>/22L-6877_A2_Ops.s3
dvc remote modify origin --local access_key_id <your-dagshub-token>
dvc remote modify origin --local secret_access_key <your-dagshub-token>
git add .dvc\config ; git commit -m "DagsHub DVC remote" ; git push
dvc push
```
`--local` writes the token to `.dvc\config.local`, which is never committed.
**Evidence:** the commits, `data\raw.dvc` contents, DagsHub showing the data. Explain: the pointer file
holds the md5 hash, size and file count of the data, not the data; Git is built for small text diffs,
so large binaries bloat every clone forever.

### Prepare the splits (needed by Task 2)
```powershell
python -m visionguard.prepare_data
git add data\splits ; git commit -m "Split lists"
```
Drops corrupt files and duplicates, converts every image once to a canonical RGB JPEG
(`data\clean\`), and writes stratified 70/15/15 lists to `data\splits\{train,val,test}.csv`.

## Task 2 - Model development

### 2.1 Input pipeline
```powershell
python -m visionguard.pipeline
```
**Evidence:** `reports\pipeline\augmented_batch.png`. The stages are documented at the top of
`src\visionguard\pipeline.py`. Image size 224 = the ImageNet resolution the pretrained weights were
trained at; aspect ratio is **kept** and the image padded to a square (stretching would distort shapes
and textures).

### 2.2 / 2.3 / 2.4 / 2.5 / 2.6 Baseline CNN and transfer learning
Before Task 3 you can train without tracking (`--tracking none`); once DagsHub is set up, the default
`--tracking manual` logs everything.
```powershell
python -m visionguard.train --model baseline --run-name baseline_cnn
python -m visionguard.train --model transfer --backbone mobilenetv2 --run-name mnv2_transfer
```
Each run writes `outputs\<time>_<name>\`:

| Sub-task | File in the run folder |
|---|---|
| 2.2 model summary, per-layer table | `model_summary.txt`, `layer_table.csv` |
| 2.3 two sets of curves, trainable layers per stage | `plots\curves_stage1_*.png`, `plots\curves_stage2_*.png`, `trainable_layers.md` |
| 2.4 curves with best epoch marked | `plots\curves_all.png` (red dashed line), `results.json` → `best_epoch` |
| 2.5 metrics, confusion matrix, ROC, threshold | `results.json`, `plots\confusion_matrix_test.png`, `plots\roc_curve.png` |
| 2.6 10 misclassified test images | `plots\misclassified_test.png`, `misclassified_test.csv` |

Design choices to explain: sigmoid + binary cross-entropy (one probability for a two-class problem),
Adam, metrics accuracy/precision/recall/AUC; seeds via `keras.utils.set_random_seed`; EarlyStopping +
ModelCheckpoint on `val_loss` + ReduceLROnPlateau. The threshold is chosen on **validation** data by
`--threshold-strategy fpr_cap --max-fpr 0.10`: at most 10% of real photos may be called AI, and within
that limit as many AI images as possible are caught - argue in the report which error is worse.

### 2.7 Keras Tuner
```powershell
python labs\keras_tuner_toy.py                 # toy: Fashion-MNIST, lr + units
python -m visionguard.tune --trials 10 --epochs 6 --tracking
```
**Evidence:** toy table from the console; `outputs\tuner\<project>\results.csv` and `importance.md`.

### Bonus - Grad-CAM
```powershell
python -m visionguard.gradcam --model outputs\<run-folder>\model.keras
```
→ `reports\gradcam.png` (4 correct, 4 wrong).

## Task 3 - MLflow and DagsHub

### 3.1 MLflow locally
```powershell
python labs\mlflow_toy.py --lr 0.001
python labs\mlflow_toy.py --lr 0.01
mlflow ui --backend-store-uri .\mlruns          # http://127.0.0.1:5000
```
Select both runs → **Compare**. Screenshot both run pages and the comparison view.

### 3.2 DagsHub
On DagsHub: repo → **Remote → Experiments** shows the tracking URL
(`https://dagshub.com/<you>/22L-6877_A2_Ops.mlflow`). Create a token under **User settings → Tokens**.
Put both in `.env` (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`).
```powershell
python labs\mlflow_toy.py --lr 0.001 --remote
git check-ignore -v .env                        # proof .env is ignored
git status --ignored                            # .env appears under "Ignored files"
```

### 3.3 / 3.4 / 3.5 Instrumented training, artifacts, comparison
```powershell
.\scripts\run_experiments.ps1                   # 8 runs, each differs from exp2 in one change
python -m visionguard.compare_runs --log        # table + chart -> reports\comparison\
```
- 3.3: `exp2_mnv2_reference` = manual logging (Keras callback), `exp8_mnv2_autolog` = MLflow autolog
  (each stage is a nested child run). Compare their run pages: params, metric curves, tags
  `git_commit`, `dvc_data_version`, `seed`.
- 3.4: Artifacts tab → `plots/` (confusion matrix, ROC), `splits/`, `environment/requirements.lock.txt`,
  `model/` (MLflow model) and `keras/model.keras`.
- 3.5: choose the best by **validation** ROC-AUC / FPR, not by test numbers.

### 3.6 Registry
```powershell
python -m visionguard.registry register --run-id <run id of 2nd best>          # version 1
python -m visionguard.registry register --run-id <run id of best> --alias champion   # version 2
python scripts\predict_champion.py data\clean\real\<some file>.jpg
```
Screenshot the registry page (two versions, `champion` alias) and the terminal output.
Then hand the champion to serving - the **only** time serving touches DagsHub:
```powershell
python -m visionguard.registry export           # -> models\champion.keras + models\champion.json
```

## Task 4 - Flask web app

```powershell
python -m visionguard.serving <real.jpg> <ai.jpg>   # 4.1 console output for two known images
python -m app                                        # http://127.0.0.1:5000, debug OFF
```
- 4.2 failure cases to screenshot: upload a `.txt` (415), a file > 5 MB (413), and a text file renamed to
  `fake.jpg` (422). Uploads are kept in RAM only (`visionguard/web.py`), file names are sanitized.
- 4.3 pages: Upload, Result, History (menu), Health.
- 4.4 `http://127.0.0.1:5000/health` shows `"debug": false`. All settings come from environment
  variables (`app/config.py`).

## Task 5 - Database trail
Every answered request inserts one row in `predictions` (time, file name + hash, decision, P(AI),
confidence, threshold, model name + registry version, local/remote source, latency). Locally it is
SQLite (`visionguard.db`); in Compose it is PostgreSQL. **History** shows it; screenshot it after a few
uploads. Optional proof straight from the database (Compose):
```powershell
docker compose exec db psql -U visionguard -d visionguard -c "select id, created_at, label, confidence, model_version from predictions order by id desc limit 5;"
```

## Task 6 - Docker and Compose

### 6.1 Warm-up - follow `labs\docker_warmup.md`.

### 6.2 Dockerfile
```powershell
docker build -t visionguard-web .
docker run --rm visionguard-web whoami           # -> appuser (non-root)
```
Points to explain: requirements copied and installed **before** the code (layer cache), `python:3.11-slim`,
`tensorflow-cpu`, non-root user, `HEALTHCHECK`, `.dockerignore` allow-list, no secrets or model inside.

### 6.3 Compose
```powershell
docker compose up -d --build
docker compose ps                                # both "healthy"
curl.exe http://127.0.0.1:5000/health
docker compose port db 5432                      # prints nothing: no published port
Test-NetConnection 127.0.0.1 -Port 5432          # TcpTestSucceeded : False
```
Model delivery: **mounted** read-only from `.\models` (swap champion = export + `docker compose restart web`).

### 6.4 Image analysis
```powershell
docker build -f Dockerfile.naive -t visionguard-web:naive .
docker images visionguard-web                    # compare the two sizes
docker history visionguard-web:latest            # which layers are largest (pip install layer)
```

### 6.5 Operations and troubleshooting
```powershell
docker compose stop ; docker compose start ; docker compose restart web
docker compose logs -f web
docker compose exec web sh       # then: whoami; ls /models; exit
docker compose exec db sh
```
Break it on purpose, diagnose **only from logs**, fix:
- **Wrong DB password:** change `POSTGRES_PASSWORD` in `.env`, `docker compose up -d`. Postgres keeps the
  password it was initialised with (in the volume), so `docker compose logs web` shows
  `password authentication failed for user "visionguard"`. Fix: restore the old value, `up -d`.
- **Missing model:** rename `models\champion.keras`, `docker compose restart web` → logs show
  `Could not load model from /models/champion.keras: Model file not found`, `/health` returns 503,
  status becomes unhealthy. Fix: rename back, restart.

### 6.6 GitHub Actions
Toy: `labs\ci_toy\README.md` (red then green run, Python 3.11 + 3.12 matrix).
Project: `.github\workflows\ci.yml` runs ruff, pytest and a Docker build on every push. Put the badge in
README.md (replace `<you>`). Secrets live in **Settings → Secrets and variables → Actions**.

## Task 7 - Model API on the cloud (Hugging Face Spaces, free CPU)
```powershell
python -m visionguard.registry export
.\deploy\prepare_hf_space.ps1 -HfUser <your-hf-username>
```
Follow the printed steps (create a Docker Space named `22L-6877_A2_Ops`, add secret `API_KEY`, push with
git-lfs). Screenshot each step and the build log. The API URL looks like
`https://<user>-22l-6877-a2-ops.hf.space`.

Research notes for the report: free CPU Space = 2 vCPU / 16 GB, sleeps after inactivity (first call is
slow), public by default (hence the API key), no persistent disk (the database stays local).
**Render** is an alternative: same `api/Dockerfile`, but the free 512 MB instance is tight for TensorFlow.

**Postman:** import `postman\VisionGuard_API.postman_collection.json`, set `baseUrl` and `apiKey`, select an
image in each predict request, run the collection. Screenshot responses + Test Results, and test once
from another network (e.g. phone hotspot).

## Task 8 - Integrate Flask with the cloud API
In `.env`: `INFERENCE_MODE=remote`, `API_URL=<Space URL>`, `API_KEY=<same secret>`, then
`python -m app` (or `docker compose up -d`). Upload several real and AI images. Each result page and the
History table show `Inference: remote:<space host>`. Screenshot results + History, and the Space logs
showing the same requests.

## Report checklist (11.1 / 11.2)
- Front matter: name, roll number (22L-6877), GitHub / DagsHub / API URLs, graded commit hash (`git rev-parse HEAD`),
  **disclosure of AI assistants used and for what** (this code was written with help from Claude).
- One chapter per task; under each sub-task ID: what, why, how (in your own words), what went wrong.
- Every screenshot captioned, referenced in the text, with username/time/project name visible.
- File name `22L-6877_VisionGuard.pdf`.
