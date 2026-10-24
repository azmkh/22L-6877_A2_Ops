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
