"""Image preprocessing shared by training AND serving.

Both sides call exactly these two functions, so a served image goes through the same steps as a
training image:

1. canonical_jpeg: any input (JPEG/PNG/WebP/GIF/BMP, RGBA, palette, EXIF-rotated) -> RGB JPEG q95.
   prepare_data.py writes this to data/clean/; the web app keeps it in memory.
2. decode_and_resize: JPEG bytes -> float32 tensor (size, size, 3) in [0, 255], aspect ratio kept
   and padded with black. Model-specific scaling lives inside the model itself.

Only Pillow is imported at module level; TensorFlow is imported lazily.
"""
import io

from PIL import Image, ImageOps

JPEG_QUALITY = 95


def _to_rgb(img: Image.Image) -> Image.Image:
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    try:
        return img.convert("RGB")
    except ValueError:  # e.g. 16-bit greyscale
        return img.convert("L").convert("RGB")


def canonical_jpeg(source) -> bytes:
    """source: a path or raw bytes. Returns RGB JPEG bytes (first frame for animated files)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)
        img = _to_rgb(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()


def decode_and_resize(jpeg_bytes, img_size: int):
    """Works on a tf.string tensor inside tf.data and on plain bytes at serving time."""
    import tensorflow as tf

    image = tf.io.decode_jpeg(jpeg_bytes, channels=3)
    image = tf.image.resize(image, (img_size, img_size), method="bilinear",
                            preserve_aspect_ratio=True, antialias=True)
    image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
    return tf.ensure_shape(image, (img_size, img_size, 3))


def decode_and_resize_np(jpeg_bytes: bytes, img_size: int):
    """NumPy/Pillow twin of decode_and_resize for the TFLite API, which has no TensorFlow.
    Same steps: scale to fit (rounded like TF), antialiased bilinear in float, centre-pad with 0."""
    import numpy as np

    with Image.open(io.BytesIO(jpeg_bytes)) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(img_size / height, img_size / width)
        new_h, new_w = int(np.round(scale * height)), int(np.round(scale * width))
        # resize each channel as a float ("F") image so no rounding to uint8 happens here
        channels = [np.asarray(c.convert("F").resize((new_w, new_h), Image.BILINEAR)) for c in img.split()]
    resized = np.stack(channels, axis=-1)
    out = np.zeros((img_size, img_size, 3), dtype=np.float32)
    top, left = (img_size - new_h) // 2, (img_size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out
