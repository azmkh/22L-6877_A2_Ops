"""Task 2.5/2.6 - threshold selection, metrics and evaluation figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DISPLAY = ["Real", "AI-generated"]


def choose_threshold(y, p, strategy="fpr_cap", max_fpr=0.10) -> float:
    """Pick the decision threshold on VALIDATION data.

    fpr_cap (default): calling a real photo "AI" is the costlier error for this product, so keep the
    false-positive rate (real -> AI) at or below max_fpr and, within that, catch as many AI images
    as possible. youden: maximise TPR - FPR. f1: maximise F1.
    """
    if strategy in ("fpr_cap", "youden"):
        fpr, tpr, thresholds = roc_curve(y, p)
        if strategy == "fpr_cap":
            allowed = np.flatnonzero(fpr <= max_fpr)
            i = allowed[np.argmax(tpr[allowed])]
        else:
            i = int(np.argmax(tpr - fpr))
        t = thresholds[i]
    elif strategy == "f1":
        precision, recall, thresholds = precision_recall_curve(y, p)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        t = thresholds[int(np.nanargmax(f1))]
    else:
        raise ValueError(strategy)
    return float(np.clip(t, 0.0, 1.0))


def compute_metrics(y, p, threshold) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "fpr_real_called_ai": fp / max(fp + tn, 1),
        "fnr_ai_called_real": fn / max(fn + tp, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def plot_confusion(y, p, threshold, path, title):
    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], DISPLAY)
    ax.set_yticks([0, 1], DISPLAY)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\nthreshold = {threshold:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(curves, threshold, path, title):
    """curves: list of (name, y, p)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, y, p in curves:
        fpr, tpr, thr = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, p):.3f})")
        pred = p >= threshold
        op_fpr = (pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        op_tpr = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        ax.scatter([op_fpr], [op_tpr], zorder=3)
    ax.plot([0, 1], [0, 1], linestyle=":", color="grey")
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title(f"{title}\ndots = operating point at threshold {threshold:.3f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history, path, title, best_epoch=None, boundaries=()):
    """history: DataFrame with an 'epoch' column (1-based) and Keras log columns."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        if metric not in history:
            continue
        ax.plot(history["epoch"], history[metric], marker="o", label=f"train {metric}")
        ax.plot(history["epoch"], history[f"val_{metric}"], marker="o", label=f"val {metric}")
        for b in boundaries:
            ax.axvline(b + 0.5, color="grey", linestyle=":", label="fine-tuning starts")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="red", linestyle="--", label=f"best epoch = {best_epoch}")
        ax.set_xlabel("epoch")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def image_grid(items, path, title, ncols=5):
    """items: list of (image path, caption)."""
    nrows = max(1, -(-len(items) // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.6 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (img_path, caption) in zip(np.ravel(axes), items):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img.thumbnail((320, 320))
            ax.imshow(img)
        ax.set_title(caption, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
