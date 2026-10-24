"""Task 1.2 - walk the raw dataset and build a per-image audit table.

For every file: class, file size, width, height, colour mode, format and SHA-256.
Files that cannot be fully decoded are flagged as corrupt; identical hashes are flagged as
duplicates (and as cross-class duplicates when the same bytes sit in both class folders).

Usage:
    python -m visionguard.data_audit            # reads data/raw, writes reports/audit/
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from visionguard import config

SKIP_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    """Open and fully decode the image so truncated files are caught, not just bad headers."""
    try:
        with Image.open(path) as img:
            info = {
                "format": img.format,
                "mode": img.mode,
                "width": img.size[0],
                "height": img.size[1],
                "n_frames": getattr(img, "n_frames", 1),
            }
            img.load()
        return {**info, "readable": True, "error": None}
    except Exception as exc:  # any decode failure means the file is unusable
        return {"format": None, "mode": None, "width": None, "height": None, "n_frames": None,
                "readable": False, "error": f"{type(exc).__name__}: {exc}"}


def display_path(path: Path) -> str:
    try:
        return path.relative_to(config.ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def audit(root: Path) -> pd.DataFrame:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower() not in SKIP_NAMES)
    rows = []
    for i, path in enumerate(files, 1):
        rel = path.relative_to(root)
        folder = rel.parts[0] if len(rel.parts) > 1 else ""
        rows.append({
            "path": display_path(path),
            "class_folder": folder,
            "label": config.label_from_folder(folder),
            "file_size": path.stat().st_size,
            "sha256": sha256_of(path),
            **inspect_image(path),
        })
        if i % 200 == 0:
            print(f"  audited {i}/{len(files)} files")
    df = pd.DataFrame(rows)
    df["is_duplicate"] = df.duplicated("sha256", keep="first")
    labels_per_hash = df.groupby("sha256")["label"].nunique()
    df["cross_class_duplicate"] = df["sha256"].map(labels_per_hash) > 1
    return df


def md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def summarize(df: pd.DataFrame) -> str:
    counts = df.groupby("label", dropna=False).agg(
        files=("path", "size"),
        readable=("readable", "sum"),
        corrupt=("readable", lambda s: int((~s).sum())),
        duplicate_copies=("is_duplicate", "sum"),
        cross_class=("cross_class_duplicate", "sum"),
    )
    counts.loc["TOTAL"] = counts.sum()

    ok = df[df["readable"]].copy()
    ok["file_size_kb"] = ok["file_size"] / 1024
    stats = ok.groupby("label")[["width", "height", "file_size_kb"]].agg(["min", "median", "max"])
    overall = ok[["width", "height", "file_size_kb"]].agg(["min", "median", "max"])
    stats.loc["ALL"] = overall.T.stack()
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.round(1)

    formats = pd.crosstab(ok["format"], ok["label"])
    modes = pd.crosstab(ok["mode"], ok["label"])
    unlabeled = df["label"].isna().sum()

    parts = [
        "# Dataset audit",
        f"Files scanned: {len(df)} | corrupt/unreadable: {int((~df['readable']).sum())} | "
        f"exact duplicate copies: {int(df['is_duplicate'].sum())} | "
        f"hashes found in both classes: {df.loc[df['cross_class_duplicate'], 'sha256'].nunique()} | "
        f"files outside a class folder: {int(unlabeled)}",
        "## Counts per class", md_table(counts),
        "## Width, height (px) and file size (KB) of readable images", md_table(stats),
        "## Formats", md_table(formats),
        "## Colour modes", md_table(modes),
    ]
    return "\n\n".join(parts) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=config.DATA_RAW)
    parser.add_argument("--out", type=Path, default=config.AUDIT_DIR)
    args = parser.parse_args(argv)

    if not args.root.exists():
        raise SystemExit(f"Dataset folder {args.root} not found. Extract the Kaggle zip there first.")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Auditing {args.root} ...")
    df = audit(args.root)
    df.to_csv(args.out / "audit.csv", index=False)
    df[~df["readable"]].to_csv(args.out / "corrupt.csv", index=False)
    dup_hashes = df.loc[df["sha256"].duplicated(keep=False), "sha256"]
    df[df["sha256"].isin(dup_hashes)].sort_values("sha256").to_csv(args.out / "duplicates.csv", index=False)
    summary = summarize(df)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote {args.out / 'audit.csv'} and summary.md")


if __name__ == "__main__":
    main()
