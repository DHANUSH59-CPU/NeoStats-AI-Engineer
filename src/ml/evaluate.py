"""Evaluation metrics and plots for the credit-risk model.

We deliberately judge the model with ranking/imbalance-aware metrics rather than
plain accuracy. With only ~8% defaults, a model that predicts "no default" for
everyone is 92% accurate but useless. ROC-AUC and PR-AUC measure how well the
model *ranks* risky applicants above safe ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Use a non-interactive backend so plots save fine inside Docker / headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils import config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Return the headline metrics as a plain dict."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # KS statistic — the standard credit-scoring separation metric: the maximum
    # gap between the cumulative true-positive and false-positive rates. Higher =
    # the model better separates defaulters from non-defaulters (0.30+ is decent).
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_statistic = float(np.max(tpr - fpr))

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks_statistic": ks_statistic,
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def save_curves(y_true, y_prob, out_dir: Path | None = None) -> dict:
    """Save ROC and PR curve PNGs; return their paths."""
    out_dir = out_dir or config.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    roc_path = out_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=110)
    plt.close()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"PR (AP={ap:.3f})")
    plt.axhline(np.mean(y_true), ls="--", color="gray",
                label=f"Base rate={np.mean(y_true):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    pr_path = out_dir / "pr_curve.png"
    plt.savefig(pr_path, dpi=110)
    plt.close()

    log.info("Saved curves: %s, %s", roc_path.name, pr_path.name)
    return {"roc_curve": str(roc_path), "pr_curve": str(pr_path)}
