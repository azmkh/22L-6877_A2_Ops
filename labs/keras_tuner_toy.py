"""Task 2.7 (toy) - tune learning rate and hidden units of a tiny network on Fashion-MNIST.

    python labs/keras_tuner_toy.py [--algo random|hyperband]
"""
import argparse
from pathlib import Path

import keras
import keras_tuner as kt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build(hp):
    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
                  "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["random", "hyperband"], default="random")
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x, x_test = x[:20000] / 255.0, x_test / 255.0
    y = y[:20000]

    common = {"objective": "val_accuracy", "directory": str(ROOT / "outputs" / "tuner_toy"),
              "project_name": args.algo, "overwrite": True, "seed": 42}
    if args.algo == "random":
        tuner = kt.RandomSearch(build, max_trials=args.trials, **common)
    else:
        tuner = kt.Hyperband(build, max_epochs=6, factor=3, **common)
    tuner.search(x, y, validation_split=0.2, epochs=4, verbose=2)

    rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
            for t in tuner.oracle.get_best_trials(num_trials=50)]
    table = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    print(table.to_string(index=False))
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best = tuner.get_best_models(1)[0]
    print("Best settings:", best_hp.values)
    print(f"Best model test accuracy: {best.evaluate(x_test, y_test, verbose=0)[1]:.4f}")


if __name__ == "__main__":
    main()
