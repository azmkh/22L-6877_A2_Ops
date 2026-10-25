"""Tasks 3.1 / 3.2 - tiny Keras model on MNIST with MLflow logging.

Local (3.1):   python labs/mlflow_toy.py --lr 0.001
               python labs/mlflow_toy.py --lr 0.01
               mlflow ui --backend-store-uri ./mlruns      -> http://127.0.0.1:5000, compare the 2 runs
DagsHub (3.2): put MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD in .env, then add --remote
"""
import argparse
import os
from pathlib import Path

import keras
import matplotlib
import mlflow
import numpy as np
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--remote", action="store_true", help="log to the MLflow server in .env (DagsHub)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    project = os.getenv("PROJECT_NAME", "22L-6877_A2_Ops")
    if args.remote:
        if not os.getenv("MLFLOW_TRACKING_URI"):
            raise SystemExit("MLFLOW_TRACKING_URI is not set - fill in .env first.")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])  # token read from env by MLflow
    else:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(f"{project}_toy")
    print("Tracking URI:", mlflow.get_tracking_uri())

    keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, y_train = x_train[:12000] / 255.0, y_train[:12000]
    x_test = x_test / 255.0

    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(args.units, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(keras.optimizers.Adam(args.lr), "sparse_categorical_crossentropy", metrics=["accuracy"])

    class LogEpoch(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + 1)

    with mlflow.start_run(run_name=f"toy_lr{args.lr:g}") as run:
        mlflow.set_tags({"kind": "toy", "dataset": "mnist-12k"})
        mlflow.log_params(vars(args))
        hist = model.fit(x_train, y_train, validation_split=0.2, epochs=args.epochs, batch_size=128,
                         callbacks=[LogEpoch()], verbose=2)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

        fig, ax = plt.subplots(figsize=(5, 3.5))
        epochs = np.arange(1, args.epochs + 1)
        ax.plot(epochs, hist.history["accuracy"], label="train")
        ax.plot(epochs, hist.history["val_accuracy"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_title(f"lr={args.lr:g}")
        ax.legend()
        mlflow.log_figure(fig, "accuracy_curve.png")
        mlflow.tensorflow.log_model(model, artifact_path="model")
        print(f"run {run.info.run_id}: test accuracy {test_acc:.4f}")


if __name__ == "__main__":
    main()
