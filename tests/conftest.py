import io

import pytest
from PIL import Image

from app import create_app


class FakePredictor:
    """Stands in for the TensorFlow model so tests run in CI without TensorFlow."""

    model_name, model_version = "fake_model", "7"

    def __init__(self, prob_ai=0.9):
        self.prob_ai = prob_ai
        self.calls = 0

    def predict(self, data, filename=None):
        self.calls += 1
        label = "ai" if self.prob_ai >= 0.5 else "real"
        return {"label": label, "probability_ai": self.prob_ai,
                "confidence": self.prob_ai if label == "ai" else 1 - self.prob_ai,
                "threshold": 0.5, "model_name": "fake_model", "model_version": "7",
                "latency_ms": 1.0, "source": "local"}

    def describe(self):
        return {"model_name": "fake_model", "model_version": "7", "source": "local"}


def image_bytes(fmt="PNG", size=(64, 48), mode="RGB"):
    buf = io.BytesIO()
    color = {"RGBA": (200, 30, 30, 128), "L": 120}.get(mode, (200, 30, 30))
    Image.new(mode, size, color).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def predictor():
    return FakePredictor()


@pytest.fixture
def app(tmp_path, predictor):
    return create_app({"DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}", "TESTING": True,
                       "MAX_UPLOAD_MB": 1}, predictor=predictor)


@pytest.fixture
def client(app):
    return app.test_client()
