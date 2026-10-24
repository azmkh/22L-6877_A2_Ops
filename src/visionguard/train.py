"""Tasks 2.2-2.6 and 3.3-3.4 - train, evaluate once on the test set, log everything to MLflow.

Transfer models train in two stages: (1) backbone frozen, new head trained; (2) last block
unfrozen and trained again with a much smaller learning rate. One ModelCheckpoint instance is
shared by both stages, so the saved file is the best epoch by validation loss over the whole run.

Examples:
    python -m visionguard.train --model baseline --run-name baseline_cnn
    python -m visionguard.train --model transfer --backbone mobilenetv2 --lr 1e-3
    python -m visionguard.train --model transfer --tracking autolog --run-name mnv2_autolog
    python -m visionguard.train --tracking none --epochs 1 --fine-tune-epochs 1   # smoke test
"""
import argparse
import contextlib
import json
import shutil
import time

import numpy as np
import pandas as pd

from visionguard import config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["baseline", "transfer"], default="transfer")
    p.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25, help="max epochs (stage 1 for transfer models)")
    p.add_argument("--fine-tune-epochs", type=int, default=15, help="max epochs of stage 2")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fine-tune-lr", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--unfreeze-layers", type=int, default=None,
                   help="stage 2: trailing backbone layers to unfreeze (default: last block, 0: no stage 2)")
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience (epochs)")
    p.add_argument("--threshold-strategy", choices=["fpr_cap", "youden", "f1"], default="fpr_cap")
    p.add_argument("--max-fpr", type=float, default=0.10,
                   help="fpr_cap: highest share of real photos we accept calling AI on validation")
    p.add_argument("--tracking", choices=["manual", "autolog", "none"], default="manual")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--deterministic", action="store_true", help="deterministic TF ops (slower)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--note", default="", help="description shown on the MLflow run page")
    return p.parse_args(argv)


def default_run_name(args):
    if args.model == "baseline":
        return f"baseline_cnn_lr{args.lr:g}_{args.img_size}"
    return f"{args.backbone}_lr{args.lr:g}_{args.img_size}{'' if args.augment else '_noaug'}"


def main(argv=None):
    args = parse_args(argv)
    import keras
    import mlflow
    import tensorflow as tf

    from visionguard import evaluate, models, pipeline, tracking

    keras.utils.set_random_seed(args.seed)  # Python, NumPy and TensorFlow seeds
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    run_name = args.run_name or default_run_name(args)
    out = config.OUTPUTS / f"{time.strftime('%Y%m%d-%H%M%S')}_{run_name}"
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    print(f"Run '{run_name}' -> {out}")

    train_ds = pipeline.make_dataset("train", args.img_size, args.batch_size, True, args.augment, args.seed)
    val_ds = pipeline.make_dataset("val", args.img_size, args.batch_size, False)
    test_ds = pipeline.make_dataset("test", args.img_size, args.batch_size, False)

    if args.model == "baseline":
        model, base = models.build_baseline_cnn(args.img_size, args.dropout), None
    else:
        model, base = models.build_transfer_model(args.backbone, args.img_size, args.dropout)
    models.compile_model(model, args.lr)
    (out / "model_summary.txt").write_text(models.summary_text(model), encoding="utf-8")
    models.layer_table(model).to_csv(out / "layer_table.csv", index=False)

    checkpoint = keras.callbacks.ModelCheckpoint(str(out / "best.keras"), monitor="val_loss", save_best_only=True)
    history_rows, boundaries, trainable_report = [], [], []

    def fit_stage(stage, epochs, initial_epoch):
        trainable = models.trainable_layer_names(model)
        n_params = models.count_params(model.trainable_weights)
        trainable_report.append(f"## {stage}: {len(trainable)} layers with weights, "
                                f"{n_params:,} trainable parameters\n" + "\n".join(trainable))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
            checkpoint,
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                              patience=max(1, args.patience // 2), min_lr=1e-7),
        ]
        if args.tracking == "manual":
            callbacks.append(tracking.epoch_logger())
        # Autolog logs fit() parameters, which would clash across two fit() calls in one run,
        # so in autolog mode each stage becomes a nested child run.
        nested = mlflow.start_run(run_name=stage, nested=True) if args.tracking == "autolog" \
            else contextlib.nullcontext()
        with nested:
            hist = model.fit(train_ds, validation_data=val_ds, epochs=initial_epoch + epochs,
                             initial_epoch=initial_epoch, callbacks=callbacks, verbose=2)
        for i in range(len(hist.history["loss"])):
            history_rows.append({"epoch": initial_epoch + i + 1, "stage": stage,
                                 **{k: float(v[i]) for k, v in hist.history.items()}})
        return initial_epoch + len(hist.history["loss"])

    params = {k: v for k, v in vars(args).items() if k not in ("note", "run_name")}
    tags = tracking.base_tags(args.seed, model_type=args.model, tracking_mode=args.tracking,
                              backbone=args.backbone if args.model == "transfer" else "none")
    with tracking.maybe_run(args.tracking != "none", run_name, params, tags, args.note) as run:
        if args.tracking == "autolog":
            tracking.enable_autolog()

        first = "baseline" if args.model == "baseline" else "stage1_frozen_backbone"
        done = fit_stage(first, args.epochs, 0)
        if args.model == "transfer" and args.unfreeze_layers != 0:
            unfrozen = models.unfreeze_backbone(base, args.backbone, args.unfreeze_layers)
            print(f"Stage 2: unfroze {len(unfrozen)} backbone layers, lr={args.fine_tune_lr:g}")
            models.compile_model(model, args.fine_tune_lr)  # recompile so trainable flags apply
            boundaries.append(done)
            stage2 = "stage2_fine_tune_last_block" if args.unfreeze_layers is None \
                else f"stage2_fine_tune_last_{args.unfreeze_layers}_layers"
            fit_stage(stage2, args.fine_tune_epochs, done)
        if args.tracking == "autolog":
            mlflow.tensorflow.autolog(disable=True)

        history = pd.DataFrame(history_rows)
        history.to_csv(out / "history.csv", index=False)
        best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
        (out / "trainable_layers.md").write_text("\n\n".join(trainable_report), encoding="utf-8")

        model = keras.models.load_model(str(out / "best.keras"))  # best epoch over all stages
        evaluate.plot_history(history, plots / "curves_all.png", f"{run_name}: all stages",
                              best_epoch, boundaries)
        for stage, part in history.groupby("stage", sort=False):
            best_in_stage = best_epoch if best_epoch in set(part["epoch"]) else None
            evaluate.plot_history(part, plots / f"curves_{stage}.png", f"{run_name}: {stage}", best_in_stage)

        # ---- evaluation: threshold from validation, test set used exactly once ----
        y_val, y_test = pipeline.labels("val"), pipeline.labels("test")
        p_val = model.predict(val_ds, verbose=0).ravel()
        p_test = model.predict(test_ds, verbose=0).ravel()
        threshold = evaluate.choose_threshold(y_val, p_val, args.threshold_strategy, args.max_fpr)
        val_m = evaluate.compute_metrics(y_val, p_val, threshold)
        test_m = evaluate.compute_metrics(y_test, p_test, threshold)
        test_m05 = evaluate.compute_metrics(y_test, p_test, 0.5)

        evaluate.plot_confusion(y_test, p_test, threshold, plots / "confusion_matrix_test.png",
                                f"{run_name} - test")
        evaluate.plot_roc([("validation", y_val, p_val), ("test", y_test, p_test)], threshold,
                          plots / "roc_curve.png", f"{run_name} - ROC")

        test_df = pipeline.read_split("test").assign(prob_ai=p_test, pred=(p_test >= threshold).astype(int))
        test_df.to_csv(out / "test_predictions.csv", index=False)
        wrong = test_df[test_df["pred"] != test_df["label"]]
        wrong = wrong.assign(margin=(wrong["prob_ai"] - threshold).abs()).sort_values("margin", ascending=False)
        wrong.to_csv(out / "misclassified_test.csv", index=False)
        if len(wrong):
            items = [(config.ROOT / r.clean_path,
                      f"true {config.CLASS_NAMES[r.label]} / pred {config.CLASS_NAMES[r.pred]}  p_ai={r.prob_ai:.2f}\n"
                      f"{r.format} {r.width:.0f}x{r.height:.0f} {r.file_size / 1024:.0f}KB")
                     for r in wrong.head(10).itertuples()]
            evaluate.image_grid(items, plots / "misclassified_test.png",
                                f"{run_name}: most confident test errors")

        model.save(str(out / "model.keras"))
        metadata = {
            "model_name": run_name, "model_type": args.model,
            "backbone": args.backbone if args.model == "transfer" else None,
            "img_size": args.img_size, "threshold": threshold,
            "threshold_strategy": args.threshold_strategy, "max_fpr": args.max_fpr,
            "class_names": config.CLASS_NAMES, "positive_class": "ai", "best_epoch": best_epoch,
            "run_id": run.info.run_id if run else None, "git_commit": tags["git_commit"],
            "dvc_data_version": tags["dvc_data_version"],
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
        results = {"threshold": threshold, "best_epoch": best_epoch, "val": val_m, "test": test_m,
                   "test_at_0.5": test_m05}
        (out / "results.json").write_text(json.dumps(results, indent=2, default=float))

        if run:
            mlflow.log_metrics({"threshold": threshold, "best_epoch": best_epoch,
                                "epochs_trained": len(history)})
            mlflow.log_metrics({f"final_val_{k}": float(v) for k, v in val_m.items()})
            mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_m.items()})
            mlflow.log_metrics({f"test05_{k}": float(v) for k, v in test_m05.items()})
            mlflow.log_artifacts(str(plots), "plots")
            for split in ("train", "val", "test"):
                mlflow.log_artifact(str(config.SPLITS_DIR / f"{split}.csv"), "splits")
            (out / "requirements.lock.txt").write_text(tracking.pip_freeze())
            mlflow.log_artifact(str(out / "requirements.lock.txt"), "environment")
            for name in ("model_summary.txt", "layer_table.csv", "trainable_layers.md", "history.csv",
                         "misclassified_test.csv", "results.json", "metadata.json"):
                mlflow.log_artifact(str(out / name), "details" if name != "metadata.json" else None)
            mlflow.log_artifact(str(out / "model.keras"), "keras")  # plain file for the serving side
            sample = np.zeros((1, args.img_size, args.img_size, 3), dtype=np.float32)
            signature = mlflow.models.infer_signature(sample, model.predict(sample, verbose=0))
            mlflow.tensorflow.log_model(model, artifact_path="model", signature=signature)
            print(f"MLflow run id: {run.info.run_id}")

    shutil.copy(out / "metadata.json", out / "model.json")  # sits next to model.keras for serving
    print(json.dumps({"threshold": threshold, "best_epoch": best_epoch, "test": test_m}, indent=2, default=float))


if __name__ == "__main__":
    main()
