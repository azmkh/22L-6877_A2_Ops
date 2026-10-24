"""Task 4.2 - upload validation for the web form."""
from werkzeug.utils import secure_filename

from visionguard.validation import ALLOWED_EXTENSIONS, InvalidImage, check_image_bytes, extension_of


class UploadError(Exception):
    def __init__(self, status, title, message):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


def validate_upload(file, max_bytes: int):
    """Return (bytes, sanitized filename) or raise UploadError. Everything stays in memory."""
    if file is None or not file.filename:
        raise UploadError(400, "No file selected", "Choose an image to upload.")
    # secure_filename strips path parts and odd characters: "../../etc/passwd.png" -> "etc_passwd.png"
    name = secure_filename(file.filename)[:200] or "upload"
    ext = extension_of(name)
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(415, "Unsupported file type",
                          f"'.{ext or '?'}' files are not accepted. Allowed: "
                          + ", ".join(sorted(ALLOWED_EXTENSIONS)) + ".")
    data = file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UploadError(413, "File too large", f"Uploads are limited to {max_bytes / 1024 / 1024:g} MB.")
    if not data:
        raise UploadError(400, "Empty file", "The uploaded file is empty.")
    try:
        check_image_bytes(data)
    except InvalidImage as exc:
        raise UploadError(422, "Unreadable image", str(exc)) from exc
    return data, name
