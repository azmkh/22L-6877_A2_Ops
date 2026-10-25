"""Task 4.4 - every setting comes from environment variables (12-factor style), so the same image
runs locally, in Compose and in the cloud with different values and no code change."""
import os
import secrets


def _float_or_none(name):
    value = os.getenv(name, "").strip()
    return float(value) if value else None


def load_settings() -> dict:
    model_path = os.getenv("MODEL_PATH", "models/champion.keras")
    return {
        "PROJECT_NAME": os.getenv("PROJECT_NAME", "22L-6877_A2_Ops"),
        "MODEL_PATH": model_path,
        "MODEL_META_PATH": os.getenv("MODEL_META_PATH") or None,
        "THRESHOLD": _float_or_none("THRESHOLD"),  # empty = use the threshold saved with the model
        "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///visionguard.db"),
        "MAX_UPLOAD_MB": float(os.getenv("MAX_UPLOAD_MB", "5")),
        "INFERENCE_MODE": os.getenv("INFERENCE_MODE", "local").lower(),  # local | remote
        "API_URL": os.getenv("API_URL", "").strip(),
        "API_KEY": os.getenv("API_KEY", ""),
        "API_TIMEOUT": float(os.getenv("API_TIMEOUT", "60")),
        "SECRET_KEY": os.getenv("SECRET_KEY") or secrets.token_hex(32),
    }
