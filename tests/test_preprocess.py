import io

import pytest
from conftest import image_bytes
from PIL import Image

from visionguard import config
from visionguard.preprocess import canonical_jpeg
from visionguard.validation import InvalidImage, check_image_bytes, extension_of


@pytest.mark.parametrize("fmt,mode", [("PNG", "RGBA"), ("PNG", "L"), ("GIF", "P"), ("WEBP", "RGB"), ("BMP", "RGB")])
def test_canonical_jpeg_is_rgb_jpeg(fmt, mode):
    out = canonical_jpeg(image_bytes(fmt, (50, 30), mode if mode != "P" else "RGB"))
    with Image.open(io.BytesIO(out)) as img:
        assert img.format == "JPEG" and img.mode == "RGB" and img.size == (50, 30)


def test_transparent_pixels_become_white():
    buf = io.BytesIO()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(buf, "PNG")
    with Image.open(io.BytesIO(canonical_jpeg(buf.getvalue()))) as img:
        assert min(img.getpixel((5, 5))) > 240


def test_check_image_bytes():
    assert check_image_bytes(image_bytes("JPEG")) == "JPEG"
    with pytest.raises(InvalidImage):
        check_image_bytes(b"GIF89a garbage")


def test_extension_of():
    assert extension_of("a.b.JPG") == "jpg"
    assert extension_of("noext") == ""


@pytest.mark.parametrize("folder,label", [("RealArt", "real"), ("AiArtData", "ai"), ("random", None)])
def test_label_from_folder(folder, label):
    assert config.label_from_folder(folder) == label


@pytest.mark.parametrize("p,t,label,conf", [
    (0.70, 0.717, "real", 0.512), (0.0, 0.717, "real", 1.0), (0.717, 0.717, "ai", 0.5),
    (1.0, 0.717, "ai", 1.0), (0.9, 0.5, "ai", 0.9), (0.2, 0.5, "real", 0.8),
])
def test_confidence_is_relative_to_threshold(p, t, label, conf):
    from visionguard.serving import decide

    got_label, got_conf = decide(p, t)
    assert got_label == label and got_conf == pytest.approx(conf, abs=1e-3)
    assert 0.5 <= got_conf <= 1.0


@pytest.mark.parametrize("size", [(300, 200), (120, 260), (224, 224), (50, 40)])
def test_numpy_resize_matches_tensorflow(size):
    """The TFLite API resizes with Pillow/NumPy; it must match the TensorFlow path used in training."""
    import numpy as np

    from visionguard.preprocess import decode_and_resize, decode_and_resize_np

    jpeg = canonical_jpeg(image_bytes("JPEG", size))
    a = decode_and_resize(jpeg, 224).numpy()
    b = decode_and_resize_np(jpeg, 224)
    assert b.shape == a.shape == (224, 224, 3)
    assert np.abs(a - b).mean() < 2.0
