"""Task 3.6 - load the registered model BY ALIAS from the MLflow registry and classify one image.

Usage:
    python scripts/predict_champion.py path/to/image.jpg [--alias champion]
"""
import argparse
import time

import mlflow
from mlflow import MlflowClient

from visionguard import config, tracking
from visionguard.preprocess import canonical_jpeg, decode_and_resize
from visionguard.serving import LABELS, decide


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--name", default=config.REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    print("Tracking URI:", tracking.setup())
    client = MlflowClient()
    mv = client.get_model_version_by_alias(args.name, args.alias)
    threshold = client.get_run(mv.run_id).data.metrics.get("threshold", 0.5)
    uri = f"models:/{args.name}@{args.alias}"
    t0 = time.perf_counter()
    model = mlflow.tensorflow.load_model(uri)
    print(f"Loaded {uri} -> version {mv.version} (run {mv.run_id}) in {time.perf_counter() - t0:.1f}s")

    import tensorflow as tf

    img_size = int(model.input_shape[1])
    with open(args.image, "rb") as fh:
        batch = decode_and_resize(tf.constant(canonical_jpeg(fh.read())), img_size)[tf.newaxis]
    prob_ai = float(model(batch, training=False).numpy()[0, 0])
    label, confidence = decide(prob_ai, threshold)
    print(f"{args.image}: {LABELS[label]}  (p_ai={prob_ai:.4f}, threshold={threshold:.3f}, "
          f"confidence={confidence:.1%})")


if __name__ == "__main__":
    main()
