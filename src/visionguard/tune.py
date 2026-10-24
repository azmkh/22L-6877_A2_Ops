"""Task 2.7 (apply) - Keras Tuner random search over learning rate, dropout and the number of
unfrozen backbone layers for the VisionGuard transfer model.

Usage:
    python -m visionguard.tune --trials 10 --epochs 6 [--tracking]
Results: outputs/tuner/<project>/results.csv and importance.md
"""
import argparse

import numpy as np
import pandas as pd

from visionguard import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trials", type=int, default=10, help="at least 8 for the assignment")
    parser.add_argument("--epochs", type=int, default=6, help="max epochs per trial")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--overwrite", action="store_true", help="start a fresh search")
    parser.add_argument("--tracking", action="store_true", help="also log the results to MLflow")
    args = parser.parse_args(argv)

    import keras
    import keras_tuner as kt

    from visionguard import models, pipeline

    keras.utils.set_random_seed(args.seed)
    train_ds = pipeline.make_dataset("train", args.img_size, args.batch_size, True, True, args.seed)
    val_ds = pipeline.make_dataset("val", args.img_size, args.batch_size, False)

    def build(hp):
        model, base = models.build_transfer_model(
            args.backbone, args.img_size, dropout=hp.Float("dropout", 0.1, 0.5, step=0.1))
        n_unfrozen = hp.Choice("unfreeze_layers", [0, 10, 20, 40])
        if n_unfrozen:
            models.unfreeze_backbone(base, args.backbone, n_unfrozen)
        models.compile_model(model, hp.Float("lr", 1e-5, 1e-3, sampling="log"))
        return model

    project = f"{config.PROJECT_NAME}_{args.backbone}"
    tuner = kt.RandomSearch(build, objective=kt.Objective("val_auc", direction="max"),
                            max_trials=args.trials, seed=args.seed, overwrite=args.overwrite,
                            directory=str(config.OUTPUTS / "tuner"), project_name=project)
    tuner.search_space_summary()
    tuner.search(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3)])

    trials = tuner.oracle.get_best_trials(num_trials=args.trials)
    df = pd.DataFrame([{"trial": t.trial_id, **t.hyperparameters.values, "val_auc": t.score} for t in trials])
    out = config.OUTPUTS / "tuner" / project
    df.to_csv(out / "results.csv", index=False)

    # Rough importance: rank correlation of each hyperparameter with the score, plus the mean
    # score per unfreeze setting. With ~10 trials this is a hint, not proof.
    ranks = df[["lr", "dropout", "unfreeze_layers", "val_auc"]].rank()
    corr = ranks.corr()["val_auc"].drop("val_auc")
    by_unfreeze = df.groupby("unfreeze_layers")["val_auc"].agg(["mean", "max", "count"])
    report = ("# Keras Tuner results\n\n" + df.to_string(index=False) +
              "\n\n## Spearman correlation with val_auc\n" + corr.round(3).to_string() +
              "\n\n## val_auc by number of unfrozen layers\n" + by_unfreeze.round(4).to_string() + "\n")
    (out / "importance.md").write_text(report, encoding="utf-8")
    print(report)
    print("Best hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)

    if args.tracking:
        import mlflow

        from visionguard import tracking

        tracking.setup()
        with mlflow.start_run(run_name=f"keras_tuner_{args.backbone}"):
            mlflow.set_tags(tracking.base_tags(args.seed, kind="keras_tuner"))
            mlflow.log_params({f"best_{k}": v for k, v in tuner.get_best_hyperparameters(1)[0].values.items()})
            mlflow.log_params({"trials": args.trials, "epochs_per_trial": args.epochs})
            mlflow.log_metric("best_val_auc", float(np.nanmax(df["val_auc"])))
            mlflow.log_artifact(str(out / "results.csv"))
            mlflow.log_artifact(str(out / "importance.md"))


if __name__ == "__main__":
    main()
