import io
import os

import pytest
from conftest import FakePredictor, image_bytes

os.environ["VISIONGUARD_API_NO_AUTOLOAD"] = "1"
from api.app import create_app  # noqa: E402


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    return create_app(predictor=FakePredictor(0.2)).test_client()


def post(client, data, name="x.png", key="s3cret"):
    headers = {"X-API-Key": key} if key else {}
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, headers=headers,
                       content_type="multipart/form-data")


def test_health_needs_no_key(api_client):
    assert api_client.get("/health").get_json()["status"] == "ok"


def test_predict_requires_key(api_client):
    assert post(api_client, image_bytes(), key=None).status_code == 401
    assert post(api_client, image_bytes(), key="wrong").status_code == 401


def test_predict_ok(api_client):
    resp = post(api_client, image_bytes())
    assert resp.status_code == 200
    assert resp.get_json()["label"] == "real"


def test_bearer_token_accepted(api_client):
    resp = api_client.post("/predict", data=image_bytes(), headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200


def test_bad_inputs(api_client):
    assert post(api_client, b"text", "a.txt").status_code == 415
    assert post(api_client, b"\xff\xd8\xff not a jpeg", "a.jpg").status_code == 422
