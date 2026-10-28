"""Task 4.1 - inference module used by the Flask web app and the model API.

The Keras model is loaded ONCE when the predictor is created (at app start-up). Loading a model
takes seconds and hundreds of MB; doing it per request would make every request slow and could
exhaust memory under concurrent load.

Preprocessing is the same code path as training (visionguard.preprocess), so the model sees
served images exactly as it saw training images.

Check it on known images:
    python -m visionguard.serving path/to/real.jpg path/to/ai.png [--model models/champion.keras]
"""
import argparse
import json
import threading
import time
from pathlib import Path

from visionguard.preprocess import canonical_jpeg, decode_and_resize

LABELS = {"ai": "AI-generated", "real": "Real"}


def decide(prob_ai: float, threshold: float):
    """Return (label, confidence). With a threshold other than 0.5, `1 - p` would be misleading
    (p=0.6 with threshold 0.7 is "Real" yet 1-p = 40%). Confidence is therefore measured from the
    threshold: 50% exactly at the decision boundary, 100% at p=0 (Real) or p=1 (AI)."""
    if prob_ai >= threshold:
        return "ai", 0.5 + 0.5 * (prob_ai - threshold) / max(1.0 - threshold, 1e-9)
    return "real", 0.5 + 0.5 * (threshold - prob_ai) / max(threshold, 1e-9)


class LocalPredictor:
    def __init__(self, model_path, meta_path=None, threshold=None):
        import keras

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        meta_path = Path(meta_path) if meta_path else model_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        self.model = keras.models.load_model(str(model_path), compile=False)
        self.img_size = int(self.model.input_shape[1])
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self.model_name = str(meta.get("registered_model") or meta.get("model_name") or model_path.stem)
        self.model_version = str(meta.get("model_version", "unregistered"))
        self._lock = threading.Lock()
        self.model.predict(self._zeros(), verbose=0)  # warm-up so the first user isn't slow

    def _zeros(self):
        import numpy as np

        return np.zeros((1, self.img_size, self.img_size, 3), dtype="float32")

    def preprocess(self, data: bytes):
        import tensorflow as tf

        image = decode_and_resize(tf.constant(canonical_jpeg(data)), self.img_size)
        return image[tf.newaxis]

    def _infer(self, batch) -> float:
        return float(self.model(batch, training=False).numpy()[0, 0])

    def predict(self, data: bytes, filename=None) -> dict:
        start = time.perf_counter()
        batch = self.preprocess(data)
        with self._lock:
            prob_ai = self._infer(batch)
        label, confidence = decide(prob_ai, self.threshold)
        return {
            "label": label,
            "label_display": LABELS[label],
            "probability_ai": round(prob_ai, 6),
            "confidence": round(confidence, 6),
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "source": "local",
        }

    def describe(self) -> dict:
        return {"model_name": self.model_name, "model_version": self.model_version,
                "threshold": self.threshold, "img_size": self.img_size, "source": "local"}


class LitePredictor(LocalPredictor):
    """Same interface for a .tflite export of the champion, run with LiteRT instead of TensorFlow.
    Used by the cloud API on small hosts: TensorFlow + Keras need ~550 MB, LiteRT a fraction."""

    def __init__(self, model_path, meta_path=None, threshold=None):
        from ai_edge_litert.interpreter import Interpreter

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        meta_path = Path(meta_path) if meta_path else model_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]
        self.img_size = int(self._input["shape"][1])
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self.model_name = str(meta.get("registered_model") or meta.get("model_name") or model_path.stem)
        self.model_version = str(meta.get("model_version", "unregistered"))
        self._lock = threading.Lock()
        self._infer(self._zeros())  # warm-up

    def _infer(self, batch) -> float:
        import numpy as np

        self.interpreter.set_tensor(self._input["index"], np.asarray(batch, dtype=np.float32))
        self.interpreter.invoke()
        return float(self.interpreter.get_tensor(self._output["index"])[0, 0])

    def preprocess(self, data: bytes):
        from visionguard.preprocess import decode_and_resize_np

        return decode_and_resize_np(canonical_jpeg(data), self.img_size)[None]


def load_predictor(model_path, meta_path=None, threshold=None):
    """LitePredictor for .tflite files, LocalPredictor (Keras) for everything else."""
    cls = LitePredictor if str(model_path).endswith(".tflite") else LocalPredictor
    return cls(model_path, meta_path, threshold)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", default="models/champion.keras")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    predictor = LocalPredictor(args.model)
    print(f"Loaded {predictor.model_name} v{predictor.model_version} in {time.perf_counter() - t0:.1f}s "
          f"(input {predictor.img_size}px, threshold {predictor.threshold:.3f})")
    for path in args.images:
        result = predictor.predict(path.read_bytes())
        print(f"{path.name}: {result['label_display']} (confidence {result['confidence']:.1%}, "
              f"p_ai={result['probability_ai']:.3f}, {result['latency_ms']} ms)")


if __name__ == "__main__":
    main()
