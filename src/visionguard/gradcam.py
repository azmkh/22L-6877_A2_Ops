"""Bonus - Grad-CAM heat maps for 4 correct and 4 wrong test predictions.

Usage:
    python -m visionguard.gradcam --model outputs/<run>/model.keras
"""
import argparse
import json
from pathlib import Path

import numpy as np

from visionguard import config


def last_conv_layer(model):
    """The last layer with a 4-D (batch, h, w, channels) output, i.e. just before global pooling."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except (AttributeError, ValueError):
            continue
    raise ValueError("no convolutional feature map found")


def heatmap(grad_model, image):
    import tensorflow as tf

    with tf.GradientTape() as tape:
        feature_maps, prob = grad_model(image[None], training=False)
        score = prob[:, 0]
    grads = tape.gradient(score, feature_maps)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # one weight per channel
    cam = tf.nn.relu(tf.reduce_sum(feature_maps[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), float(prob[0, 0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4, help="images per group (correct / wrong)")
    parser.add_argument("--out", type=Path, default=config.REPORTS / "gradcam.png")
    args = parser.parse_args(argv)

    import keras
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    from visionguard import pipeline
    from visionguard.preprocess import decode_and_resize

    model = keras.models.load_model(str(args.model))
    metas = [m for m in (args.model.with_suffix(".json"), args.model.with_name("metadata.json")) if m.exists()]
    threshold = json.loads(metas[0].read_text())["threshold"] if metas else 0.5
    img_size = int(model.input_shape[1])
    layer = last_conv_layer(model)
    grad_model = keras.Model(model.inputs, [layer.output, model.output])
    print(f"Grad-CAM on layer '{layer.name}', threshold {threshold:.3f}")

    df = pipeline.read_split("test")
    images = [decode_and_resize(tf.io.read_file(str(config.ROOT / p)), img_size) for p in df["clean_path"]]
    probs = model.predict(tf.stack(images), verbose=0).ravel()
    correct = np.flatnonzero((probs >= threshold) == (df["label"].to_numpy() == 1))
    wrong = np.setdiff1d(np.arange(len(df)), correct)
    rng = np.random.default_rng(config.SEED)
    picks = [("correct", i) for i in rng.permutation(correct)[: args.n]] + \
            [("wrong", i) for i in rng.permutation(wrong)[: args.n]]

    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6))
    for col, (group, i) in enumerate(picks):
        cam, p = heatmap(grad_model, images[i])
        cam = tf.image.resize(cam[..., None], (img_size, img_size)).numpy()[..., 0]
        rgb = np.clip(images[i].numpy(), 0, 255).astype("uint8")
        truth = config.CLASS_NAMES[int(df['label'].iloc[i])]
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"{group}\ntrue {truth}, p_ai={p:.2f}", fontsize=8)
        axes[1, col].imshow(rgb)
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM ({layer.name}) - warm colours = regions that pushed the score")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
