"""Task 7 - WSGI entry point for the model API on PythonAnywhere (free plan, no card needed).

Paste this file's contents into the WSGI configuration file linked on the PythonAnywhere Web tab
(/var/www/<username>_pythonanywhere_com_wsgi.py). It serves the same api/app.py as the Docker images,
running the TFLite champion with LiteRT (no TensorFlow: the free plan has 512 MB of disk).

Expects, in the PythonAnywhere home folder:
    ~/22L-6877_A2_Ops            git clone of this repository
    ~/.visionguard_api_key       the API key on one line (chmod 600), never committed
"""
import os
import sys
from pathlib import Path

REPO = Path.home() / "22L-6877_A2_Ops"
sys.path[:0] = [str(REPO), str(REPO / "src")]

os.environ.setdefault("MODEL_PATH", str(REPO / "deploy" / "render" / "champion.tflite"))
os.environ.setdefault("API_KEY", (Path.home() / ".visionguard_api_key").read_text().strip())

from api.app import app as application  # noqa: E402, F401  (WSGI servers look for "application")
