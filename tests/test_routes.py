import io

from conftest import image_bytes


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["model_loaded"] is True
    assert body["debug"] is False


def test_health_degraded_without_model(tmp_path):
    from app import create_app

    app = create_app({"DATABASE_URL": f"sqlite:///{tmp_path / 'x.db'}", "INFERENCE_MODE": "remote",
                      "API_URL": ""})
    resp = app.test_client().get("/health")
    assert resp.status_code == 503
    assert "API_URL" in resp.get_json()["error"]


def test_valid_upload_is_predicted_and_logged(client, app, predictor):
    resp = upload(client, image_bytes(), "cat photo.png")
    assert resp.status_code == 302
    page = client.get(resp.headers["Location"])
    assert page.status_code == 200
    assert b"AI-generated" in page.data and b"90.0%" in page.data
    rows = app.extensions["db"].recent()
    assert len(rows) == 1
    row = rows[0]
    assert row.filename == "cat_photo.png"  # sanitized
    assert row.model_name == "fake_model" and row.model_version == "7"
    assert predictor.calls == 1


def test_history_lists_predictions(client):
    upload(client, image_bytes(), "one.png")
    upload(client, image_bytes("JPEG"), "two.jpg")
    page = client.get("/history")
    assert page.status_code == 200
    assert b"one.png" in page.data and b"two.jpg" in page.data
    assert b"fake_model" in page.data
    assert b"one.png" not in client.get("/history?label=real").data


def test_rejects_wrong_file_type(client, predictor):
    resp = upload(client, b"hello", "notes.txt")
    assert resp.status_code == 415
    assert b"Unsupported file type" in resp.data
    assert predictor.calls == 0


def test_rejects_file_over_limit(client, predictor):
    resp = upload(client, b"\xff\xd8" + b"0" * (2 * 1024 * 1024), "huge.jpg")
    assert resp.status_code == 413
    assert predictor.calls == 0


def test_rejects_fake_image(client, predictor):
    resp = upload(client, b"\x89PNG\r\n\x1a\n this is not really a png", "fake.png")
    assert resp.status_code == 422
    assert b"could not be decoded" in resp.data
    assert predictor.calls == 0


def test_rejects_truncated_image(client):
    data = image_bytes("JPEG", (400, 400))
    resp = upload(client, data[: len(data) // 2], "cut.jpg")
    assert resp.status_code == 422


def test_path_traversal_name_is_sanitized(client, app):
    upload(client, image_bytes(), "../../etc/passwd.png")
    assert app.extensions["db"].recent()[0].filename == "etc_passwd.png"


def test_missing_file(client):
    resp = client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_unknown_result_is_404(client):
    assert client.get("/result/999").status_code == 404
