"""Task 8 - predictor that calls the cloud model API instead of running TensorFlow locally."""
import time
from urllib.parse import urlparse

import requests

LABELS = {"ai": "AI-generated", "real": "Real"}


class PredictorError(RuntimeError):
    pass


class RemotePredictor:
    def __init__(self, api_url: str, api_key: str, timeout: float = 60):
        self.api_url = api_url.rstrip("/")
        self.host = urlparse(self.api_url).netloc or self.api_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key
        self.last_model = {}

    def predict(self, data: bytes, filename=None) -> dict:
        start = time.perf_counter()
        try:
            resp = self.session.post(f"{self.api_url}/predict", timeout=self.timeout,
                                     files={"image": (filename or "upload.jpg", data)})
        except requests.RequestException as exc:
            raise PredictorError(f"Model API at {self.host} is unreachable: {exc}") from exc
        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text[:200]
            raise PredictorError(f"Model API returned HTTP {resp.status_code}: {detail}")
        result = resp.json()
        result["server_latency_ms"] = result.get("latency_ms")
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)  # full round trip
        result["source"] = f"remote:{self.host}"
        result.setdefault("label_display", LABELS.get(result["label"], result["label"]))
        self.last_model = {k: result.get(k) for k in ("model_name", "model_version", "threshold")}
        return result

    def describe(self) -> dict:
        # The health check deliberately does not call the API: the local app stays healthy even
        # if the cloud service is asleep, and /health stays fast.
        return {"source": "remote", "api_url": self.api_url, **self.last_model}
