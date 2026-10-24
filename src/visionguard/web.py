"""Flask helpers shared by the web app and the model API."""
import io

from flask import Request


class InMemoryRequest(Request):
    """Werkzeug spools uploads larger than 500 KB to a temporary file. Returning a BytesIO keeps
    every upload in RAM only, so an uploaded image never touches the disk. The request size cap
    (MAX_CONTENT_LENGTH) bounds how much memory this can use."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()
