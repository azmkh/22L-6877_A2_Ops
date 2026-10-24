"""Task 3.6 - model registry helpers, and the hand-off of the champion model to serving.

    python -m visionguard.registry register --run-id <id> [--alias champion]
    python -m visionguard.registry alias --version 2 --alias champion
    python -m visionguard.registry export [--alias champion]

`export` is the ONLY step that talks to DagsHub on the serving side: it downloads the champion's
model.keras + metadata.json once into models/. The web app and the API then load that local file
at start-up, so no request ever waits on DagsHub.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from visionguard import config, tracking


def register(run_id, name, alias=None):
    version = mlflow.register_model(f"runs:/{run_id}/model", name)
    client = MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    for key in ("threshold", "test_roc_auc", "test_f1"):
        if key in metrics:
            client.set_model_version_tag(name, version.version, key, f"{metrics[key]:.4f}")
    if alias:
        client.set_registered_model_alias(name, alias, version.version)
    print(f"Registered {name} version {version.version} from run {run_id}"
          + (f" with alias '{alias}'" if alias else ""))


def set_alias(name, version, alias):
    MlflowClient().set_registered_model_alias(name, alias, str(version))
    print(f"{name}@{alias} -> version {version}")


def export(name, alias, out_dir: Path):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        model_file = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras/model.keras",
                                                         dst_path=tmp)
        meta_file = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="metadata.json",
                                                        dst_path=tmp)
        shutil.copy(model_file, out_dir / "champion.keras")
        meta = json.loads(Path(meta_file).read_text())
    meta.update({"registered_model": name, "alias": alias, "model_version": mv.version, "run_id": mv.run_id})
    (out_dir / "champion.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {name}@{alias} (version {mv.version}, run {mv.run_id}) to {out_dir / 'champion.keras'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=config.REGISTERED_MODEL_NAME)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_reg = sub.add_parser("register")
    p_reg.add_argument("--run-id", required=True)
    p_reg.add_argument("--alias")
    p_alias = sub.add_parser("alias")
    p_alias.add_argument("--version", required=True)
    p_alias.add_argument("--alias", default="champion")
    p_exp = sub.add_parser("export")
    p_exp.add_argument("--alias", default="champion")
    p_exp.add_argument("--out", type=Path, default=config.MODELS_DIR)
    args = parser.parse_args(argv)

    tracking.setup()
    if args.cmd == "register":
        register(args.run_id, args.name, args.alias)
    elif args.cmd == "alias":
        set_alias(args.name, args.version, args.alias)
    else:
        export(args.name, args.alias, args.out)


if __name__ == "__main__":
    main()
