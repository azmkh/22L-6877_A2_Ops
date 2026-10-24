"""Run the app with Flask's built-in server, debug OFF: `python -m app`.

Debug mode must stay off outside development: the Werkzeug debugger lets anyone who can reach an
error page execute Python on the server, it leaks stack traces and settings, and the reloader
loads the model twice. Inside Docker the app runs under gunicorn instead (see Dockerfile)."""
import os

from dotenv import load_dotenv

load_dotenv()

from app import create_app  # noqa: E402

if __name__ == "__main__":
    create_app().run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
