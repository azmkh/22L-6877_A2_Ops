"""Task 1.3 - visual EDA from the audit table.

Writes to reports/eda/: a grid of sample images per class, histograms of width, height and file
size per class, and a bar chart of formats/colour modes.

Usage:
    python -m visionguard.eda [--per-class 8]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from visionguard import config  # noqa: E402


def sample_grid(df, per_class, seed, path):
    classes = sorted(df["label"].dropna().unique())
    fig, axes = plt.subplots(len(classes), per_class, figsize=(2.2 * per_class, 2.6 * len(classes)))
    axes = np.atleast_2d(axes)
    for r, label in enumerate(classes):
        rows = df[df["label"] == label]
        picks = rows.sample(min(per_class, len(rows)), random_state=seed)
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c >= len(picks):
                continue
            row = picks.iloc[c]
            with Image.open(config.ROOT / row["path"]) as img:
                img = img.convert("RGB")
                img.thumbnail((256, 256))
                ax.imshow(img)
            ax.set_title(f"{row['width']}x{row['height']} {row['format']}", fontsize=8)
        axes[r][0].text(-0.15, 0.5, label.upper(), transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=12, weight="bold")
    fig.suptitle("Random samples per class")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def histograms(df, path):
    df = df.assign(file_size_kb=df["file_size"] / 1024)
    cols = [("width", "Width (px)"), ("height", "Height (px)"), ("file_size_kb", "File size (KB)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, title) in zip(axes, cols):
        upper = df[col].quantile(0.99)  # clip the extreme tail so the bulk stays readable
        bins = np.linspace(0, upper, 40)
        for label, grp in df.groupby("label"):
            ax.hist(grp[col].clip(upper=upper), bins=bins, alpha=0.55, label=label)
            ax.axvline(grp[col].median(), linestyle="--", linewidth=1, color=ax.patches[-1].get_facecolor())
        ax.set_title(f"{title} (dashed = median, clipped at p99)")
        ax.set_xlabel(title)
        ax.set_ylabel("images")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def formats_and_modes(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pd.crosstab(df["format"], df["label"]).plot.bar(ax=axes[0], title="File format per class")
    pd.crosstab(df["mode"], df["label"]).plot.bar(ax=axes[1], title="Colour mode per class")
    for ax in axes:
        ax.set_ylabel("images")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)

    df = pd.read_csv(config.AUDIT_DIR / "audit.csv")
    df = df[df["readable"] & df["label"].notna()]
    config.EDA_DIR.mkdir(parents=True, exist_ok=True)
    sample_grid(df, args.per_class, args.seed, config.EDA_DIR / "samples_per_class.png")
    histograms(df, config.EDA_DIR / "histograms.png")
    formats_and_modes(df, config.EDA_DIR / "formats_modes.png")
    print(f"Figures written to {config.EDA_DIR}")


if __name__ == "__main__":
    main()
