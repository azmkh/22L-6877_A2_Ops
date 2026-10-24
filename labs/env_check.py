"""Task 1.1 - print exact library versions and whether TensorFlow sees a GPU.
Run inside the activated .venv and screenshot the output (it prints user, host and time)."""
import getpass
import importlib.metadata as md
import platform
import socket
import sys
from datetime import datetime

PACKAGES = ["tensorflow", "keras", "keras-tuner", "mlflow", "dvc", "numpy", "pandas", "scikit-learn",
            "matplotlib", "pillow", "flask", "sqlalchemy"]

print(f"user={getpass.getuser()} host={socket.gethostname()} time={datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"python {sys.version.split()[0]} ({sys.executable})")
print(f"platform {platform.platform()}")
for name in PACKAGES:
    try:
        print(f"  {name:<14} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name:<14} (not installed)")

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPUs visible to TensorFlow: {gpus or 'none'}")
if not gpus and platform.system() == "Windows":
    print("Note: TensorFlow >= 2.11 has no GPU support on native Windows; use WSL2 for a GPU.")
