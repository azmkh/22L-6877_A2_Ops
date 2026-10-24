"""Task 2.1 - tf.data input pipeline.

Stages (in order):
  1. from_tensor_slices   the (path, label) pairs from data/splits/<split>.csv
  2. map(load)            read file -> decode JPEG -> resize keeping aspect ratio -> pad to square
  3. cache                decoded images stay in RAM after the first epoch (dataset is small)
  4. shuffle              training split only, reshuffled every epoch, seeded
  5. batch
  6. map(augment)         training split only: flip, small rotation/zoom, contrast
  7. prefetch             overlaps preprocessing of the next batch with training on this one

Run `python -m visionguard.pipeline` to save a figure of one batch before/after augmentation.
"""
import argparse

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from visionguard import config
from visionguard.preprocess import decode_and_resize

AUTOTUNE = tf.data.AUTOTUNE


def read_split(split: str) -> pd.DataFrame:
    path = config.SPLITS_DIR / f"{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m visionguard.prepare_data` first.")
    return pd.read_csv(path)


def labels(split: str) -> np.ndarray:
    return read_split(split)["label"].to_numpy(dtype=np.int64)


def make_augmenter(seed: int = config.SEED) -> keras.Sequential:
    # Mild, geometry/colour-only augmentation. No blur, noise or re-compression: those would
    # erase exactly the fine texture/frequency cues that separate generated from real images.
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.05, fill_mode="constant", seed=seed),
        keras.layers.RandomZoom(0.1, fill_mode="constant", seed=seed),
        keras.layers.RandomContrast(0.1, seed=seed),
    ], name="augment")


def make_dataset(split, img_size, batch_size, training, augment=True, seed=config.SEED, cache=True):
    df = read_split(split)
    paths = [str(config.ROOT / p) for p in df["clean_path"]]
    y = df["label"].to_numpy(dtype=np.float32).reshape(-1, 1)  # (N, 1) matches the sigmoid output

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.map(lambda p, t: (decode_and_resize(tf.io.read_file(p), img_size), t),
                num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augment:
        augmenter = make_augmenter(seed)
        ds = ds.map(lambda x, t: (augmenter(x, training=True), t), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def save_augmented_batch_figure(path, img_size=224, n=8, seed=config.SEED):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keras.utils.set_random_seed(seed)
    # shuffled but not yet augmented, so both classes appear; then augment the same images
    images, ys = next(iter(make_dataset("train", img_size, n, training=True, augment=False, cache=False)))
    augmented = make_augmenter(seed)(images, training=True)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5.8))
    for i in range(n):
        name = config.CLASS_NAMES[int(ys[i, 0])]
        for row, batch, title in [(0, images, "original"), (1, augmented, "augmented")]:
            axes[row, i].imshow(np.clip(batch[i].numpy(), 0, 255).astype("uint8"))
            axes[row, i].set_title(f"{title}\n{name}", fontsize=8)
            axes[row, i].axis("off")
    fig.suptitle(f"One training batch, {img_size}x{img_size}, aspect ratio kept + black padding")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args(argv)
    out = config.REPORTS / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    save_augmented_batch_figure(out / "augmented_batch.png", args.img_size)
    ds = make_dataset("train", args.img_size, 32, training=True)
    print("element_spec:", ds.element_spec)
    print(f"Figure written to {out / 'augmented_batch.png'}")


if __name__ == "__main__":
    main()
