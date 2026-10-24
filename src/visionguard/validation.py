"""Upload checks shared by the web app and the model API (Pillow only, no TensorFlow)."""
import io

from PIL import Image

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP", "GIF"}
MAX_PIXELS = 40_000_000  # refuse decompression bombs before decoding them


class InvalidImage(ValueError):
    pass


def extension_of(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def check_image_bytes(data: bytes) -> str:
    """Raise InvalidImage unless the bytes fully decode as a supported image. Returns the format."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            if fmt not in ALLOWED_FORMATS:
                raise InvalidImage(f"Image format {fmt} is not supported.")
            width, height = img.size
            if width * height > MAX_PIXELS:
                raise InvalidImage(f"Image is too large ({width}x{height} pixels).")
            img.load()
    except InvalidImage:
        raise
    except Exception as exc:
        raise InvalidImage("The file looks like an image but could not be decoded.") from exc
    return fmt
