"""Tasks 2.2 and 2.3 - the baseline CNN and the transfer-learning model.

Both models take float32 images in [0, 255] and do their own scaling as their first layer, so the
serving code never has to know which backbone was trained.
"""
import keras
import pandas as pd
from keras import layers

# rescale: the scaling the backbone's ImageNet weights expect, or None if the backbone already
# contains it (EfficientNet in Keras ships its own Rescaling + Normalization layers).
# last_block: name of the first layer of the final block, unfrozen in fine-tuning stage 2.
BACKBONES = {
    "mobilenetv2": {"cls": keras.applications.MobileNetV2, "rescale": (1 / 127.5, -1.0),
                    "last_block": "block_16_expand"},
    "efficientnetb0": {"cls": keras.applications.EfficientNetB0, "rescale": None,
                       "last_block": "block7a_expand_conv"},
}


def build_baseline_cnn(img_size: int, dropout: float = 0.3) -> keras.Model:
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale_0_1")(inputs)
    for i, filters in enumerate([32, 64, 128], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False, name="last_conv")(x)
    x = layers.BatchNormalization(name="bn_last")(x)
    x = layers.ReLU(name="relu_last")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(backbone: str, img_size: int, dropout: float = 0.3, weights="imagenet"):
    """Returns (model, base). The backbone is built on our input tensor (not nested as a single
    layer), so every backbone layer is visible in the graph - this keeps Grad-CAM simple."""
    spec = BACKBONES[backbone]
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = inputs
    if spec["rescale"]:
        scale, offset = spec["rescale"]
        x = layers.Rescaling(scale, offset=offset, name="backbone_preprocess")(x)
    base = spec["cls"](include_top=False, weights=weights, input_tensor=x, input_shape=(img_size, img_size, 3))
    base.trainable = False
    h = layers.GlobalAveragePooling2D(name="head_gap")(base.output)
    h = layers.Dropout(dropout, name="head_dropout")(h)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(h)
    return keras.Model(inputs, outputs, name=f"{backbone}_transfer"), base


def unfreeze_backbone(base: keras.Model, backbone: str, n_layers=None) -> list:
    """Unfreeze the last block (n_layers=None) or the last n_layers backbone layers.
    BatchNorm layers stay frozen: with trainable=False they also run in inference mode, so their
    ImageNet statistics are not wrecked by small batches. Returns the unfrozen layer names."""
    backbone_layers = [lyr for lyr in base.layers if not isinstance(lyr, keras.layers.InputLayer)]
    if n_layers is None:
        names = [lyr.name for lyr in backbone_layers]
        start = names.index(BACKBONES[backbone]["last_block"])
    else:
        start = max(0, len(backbone_layers) - n_layers)
    base.trainable = True
    for i, layer in enumerate(backbone_layers):
        layer.trainable = i >= start and not isinstance(layer, keras.layers.BatchNormalization)
    return [lyr.name for lyr in backbone_layers[start:] if lyr.trainable]


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def trainable_layer_names(model: keras.Model) -> list:
    return [lyr.name for lyr in model.layers if lyr.trainable and lyr.trainable_weights]


def count_params(weights) -> int:
    total = 0
    for w in weights:
        n = 1
        for dim in w.shape:
            n *= int(dim)
        total += n
    return total


def layer_table(model: keras.Model) -> pd.DataFrame:
    rows = []
    for layer in model.layers:
        try:
            shape = tuple(layer.output.shape)
        except (AttributeError, ValueError):
            shape = None
        rows.append({"layer": layer.name, "type": type(layer).__name__, "output_shape": shape,
                     "params": layer.count_params(), "trainable": layer.trainable})
    return pd.DataFrame(rows)


def summary_text(model: keras.Model) -> str:
    lines = []
    model.summary(line_length=110, print_fn=lambda *a, **k: lines.append(str(a[0]) if a else ""))
    return "\n".join(lines)
