"""Shared settings for the training-side scripts. Everything can be overridden from .env."""
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# On Windows, output redirected to a file/pipe defaults to cp1252, and MLflow prints an emoji
# ("View run ...") when a run ends on a remote server, which crashed the script at the last step.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# The assignment asks for "<RollNumber>_A2_Ops" everywhere (roll number 22L-6877).
PROJECT_NAME = os.getenv("PROJECT_NAME", "22L-6877_A2_Ops")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", f"{PROJECT_NAME}_visionguard")
SEED = int(os.getenv("SEED", "42"))

DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
SPLITS_DIR = ROOT / "data" / "splits"
REPORTS = ROOT / "reports"
AUDIT_DIR = REPORTS / "audit"
EDA_DIR = REPORTS / "eda"
OUTPUTS = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

# Index in this list is the integer label. AI-generated is the positive class (label 1),
# so the sigmoid output is P(image is AI-generated).
CLASS_NAMES = ["real", "ai"]


def label_from_folder(folder: str):
    """Map a dataset folder name (e.g. 'RealArt', 'AiArtData') to a class name."""
    name = folder.lower()
    if "real" in name:
        return "real"
    if "ai" in name or "fake" in name or "generated" in name:
        return "ai"
    return None


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit():
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty():
    status = _git("status", "--porcelain")
    return "unknown" if status is None else str(bool(status))


def dvc_data_version():
    """The md5 that DVC recorded for data/raw (the pointer file), i.e. the dataset version."""
    pointer = ROOT / "data" / "raw.dvc"
    if not pointer.exists():
        return "untracked"
    import yaml

    meta = yaml.safe_load(pointer.read_text())
    return str(meta["outs"][0].get("md5", "unknown"))
