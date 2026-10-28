"""Task 7 (Render) - convert the exported champion to TFLite for the small-memory cloud API.

Run after `python -m visionguard.registry export`:
    python scripts/export_tflite.py [--model models/champion.keras] [--deploy deploy/render]

Writes models/champion.tflite and copies it, with champion.json, into the Render deploy folder,
which is committed so Render can build the image straight from GitHub.
"""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/champion.keras", type=Path)
    parser.add_argument("--deploy", default="deploy/render", type=Path)
    args = parser.parse_args()

    import keras
    import tensorflow as tf

    model = keras.models.load_model(str(args.model), compile=False)
    tflite = args.model.with_suffix(".tflite")
    tflite.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
    print(f"Wrote {tflite} ({tflite.stat().st_size / 2**20:.1f} MB)")

    args.deploy.mkdir(parents=True, exist_ok=True)
    for src in (tflite, args.model.with_suffix(".json")):
        shutil.copy2(src, args.deploy / src.name)
        print(f"Copied {src} -> {args.deploy / src.name}")


if __name__ == "__main__":
    main()
