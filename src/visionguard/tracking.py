"""Task 3 - MLflow helpers. The tracking address and token come only from environment variables
(MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD), loaded from .env."""
import contextlib
import os
import subprocess
import sys

import mlflow

from visionguard import config


def setup(experiment=None) -> str:
    """Without MLFLOW_TRACKING_URI, MLflow falls back to the local ./mlruns folder."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", config.PROJECT_NAME))
    return mlflow.get_tracking_uri()


def base_tags(seed, **extra) -> dict:
    return {
        "project": config.PROJECT_NAME,
        "git_commit": config.git_commit(),
        "git_dirty": config.git_dirty(),
        "dvc_data_version": config.dvc_data_version(),
        "seed": str(seed),
        **{k: str(v) for k, v in extra.items()},
    }


@contextlib.contextmanager
def maybe_run(enabled, run_name, params, tags, note=""):
    if not enabled:
        yield None
        return
    print("MLflow tracking URI:", setup())
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_params(params)
        yield run


def enable_autolog():
    """MLflow's TensorFlow autologging. Models are logged by us (once, after evaluation), and
    per-epoch checkpoints are switched off so nothing large is uploaded every epoch."""
    try:
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, checkpoint=False)
    except TypeError:  # older MLflow without the checkpoint option
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False)


def epoch_logger():
    """Manual alternative to autolog: a Keras callback that logs every epoch's metrics."""
    import keras

    class MLflowEpochLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            metrics = {k: float(v) for k, v in (logs or {}).items()}
            lr = self.model.optimizer.learning_rate
            metrics["lr"] = float(keras.ops.convert_to_numpy(lr))
            # `epoch` already continues across stages because stage 2 uses initial_epoch.
            mlflow.log_metrics(metrics, step=epoch + 1)

    return MLflowEpochLogger()


def pip_freeze() -> str:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        return out.stdout
    # venvs made by uv have no pip: list the installed distributions directly
    import importlib.metadata as md

    pkgs = {d.metadata["Name"]: d.version for d in md.distributions() if d.metadata["Name"]}
    return "".join(f"{name}=={pkgs[name]}\n" for name in sorted(pkgs, key=str.lower))
