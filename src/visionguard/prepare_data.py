"""Clean the audited dataset and write stratified train/val/test split lists.

Drops corrupt files, extra copies of exact duplicates, and every copy of an image that appears in
both classes (its label is ambiguous). Each kept image is converted once to a canonical RGB JPEG
(see preprocess.py) under data/clean/<class>/<sha>.jpg, so tf.data only has to decode JPEG.
Splitting after de-duplication guarantees no identical image lands in two splits.

Usage:
    python -m visionguard.prepare_data [--val 0.15 --test 0.15]
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from visionguard import config
from visionguard.preprocess import canonical_jpeg

SPLIT_COLUMNS = ["clean_path", "label", "class", "sha256", "orig_path", "width", "height",
                 "format", "mode", "file_size"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)

    df = pd.read_csv(config.AUDIT_DIR / "audit.csv")
    steps = [
        ("not readable", ~df["readable"]),
        ("outside a class folder", df["label"].isna()),
        ("same image in both classes", df["cross_class_duplicate"]),
        ("extra copy of a duplicate", df["is_duplicate"]),
    ]
    keep = pd.Series(True, index=df.index)
    for reason, mask in steps:
        dropped = int((keep & mask).sum())
        keep &= ~mask
        print(f"dropped {dropped:5d}  ({reason})")
    df = df[keep].copy()

    clean_paths, failed = [], []
    for row in df.itertuples():
        out = config.DATA_CLEAN / row.label / f"{row.sha256[:16]}.jpg"
        try:
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(canonical_jpeg(config.ROOT / row.path))
            clean_paths.append(out.relative_to(config.ROOT).as_posix())
        except Exception as exc:
            failed.append(row.Index)
            clean_paths.append(None)
            print(f"  could not convert {row.path}: {exc}")
    df["clean_path"] = clean_paths
    df = df.drop(index=failed)
    df = df.rename(columns={"path": "orig_path", "label": "class"})
    df["label"] = df["class"].map(config.CLASS_NAMES.index)

    train, rest = train_test_split(df, test_size=args.val + args.test, stratify=df["label"],
                                   random_state=args.seed)
    val, test = train_test_split(rest, test_size=args.test / (args.val + args.test),
                                 stratify=rest["label"], random_state=args.seed)

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.sort_values("clean_path")[SPLIT_COLUMNS]
        part.to_csv(config.SPLITS_DIR / f"{name}.csv", index=False)
        counts = part["class"].value_counts().to_dict()
        print(f"{name:5s}: {len(part):5d} images  {counts}")
    print(f"Split lists written to {config.SPLITS_DIR}")


if __name__ == "__main__":
    main()
