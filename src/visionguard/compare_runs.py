"""Task 3.5 - pull every top-level run of the experiment into one comparison table and chart.

Select the winner by VALIDATION metrics; the test numbers are reported, not used for choosing,
otherwise the test set silently becomes a second validation set.

Usage:
    python -m visionguard.compare_runs [--log]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from visionguard import config, tracking  # noqa: E402

PARAMS = ["model", "backbone", "lr", "img_size", "augment", "unfreeze_layers", "tracking"]
METRICS = ["final_val_roc_auc", "final_val_f1", "test_roc_auc", "test_f1", "test_precision",
           "test_recall", "test_fpr_real_called_ai", "best_epoch", "threshold"]
CONFIG_PARAMS = ["model", "backbone", "lr", "img_size", "batch_size", "augment",
                 "unfreeze_layers", "fine_tune_lr", "dropout", "seed", "tracking"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="store_true", help="log the table and chart as a new MLflow run")
    args = parser.parse_args(argv)

    tracking.setup()
    runs = mlflow.search_runs(experiment_names=[config.PROJECT_NAME], order_by=["start_time ASC"])
    if "tags.mlflow.parentRunId" in runs:
        runs = runs[runs["tags.mlflow.parentRunId"].isna()]  # drop autolog child runs
    runs = runs[runs.get("metrics.test_roc_auc").notna()] if "metrics.test_roc_auc" in runs else runs.iloc[0:0]
    if runs.empty:
        raise SystemExit("No finished training runs found in the experiment.")

    # Repeated executions of the same configuration are useful for reproducibility checks,
    # but they are not distinct experiment conditions. Keep the newest completed run for
    # each configuration so reruns cannot inflate the comparison count.
    signature = [f"params.{name}" for name in CONFIG_PARAMS if f"params.{name}" in runs]
    if signature:
        runs = runs.sort_values("start_time").drop_duplicates(subset=signature, keep="last")

    cols = {"tags.mlflow.runName": "run"}
    cols.update({f"params.{p}": p for p in PARAMS if f"params.{p}" in runs})
    cols.update({f"metrics.{m}": m for m in METRICS if f"metrics.{m}" in runs})
    table = runs[list(cols)].rename(columns=cols).round(4)
    if {"model", "backbone"}.issubset(table.columns):
        table.loc[table["model"] == "baseline", "backbone"] = "none"
    table = table.sort_values("final_val_roc_auc", ascending=False)

    out = config.REPORTS / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "runs.csv", index=False)
    (out / "runs.md").write_text(table.to_string(index=False), encoding="utf-8")
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(table)), 4.5))
    x = range(len(table))
    width = 0.27
    for k, metric in enumerate(["final_val_roc_auc", "test_roc_auc", "test_f1"]):
        if metric in table:
            ax.bar([i + (k - 1) * width for i in x], table[metric], width, label=metric)
    ax.set_xticks(list(x), table["run"], rotation=30, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Runs sorted by validation ROC-AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "runs.png", dpi=120)
    print(f"Best by validation ROC-AUC: {table.iloc[0]['run']}  (table + chart in {out})")

    if args.log:
        with mlflow.start_run(run_name="run_comparison"):
            mlflow.set_tag("kind", "comparison")
            mlflow.log_artifacts(str(out), "comparison")


if __name__ == "__main__":
    main()
