"""Explainable AI using SHAP.

SHAP attributes a single prediction to its features: how many "log-odds" each
feature pushed the default risk up (positive) or down (negative). This is what
turns a black-box score into an auditable, regulator-friendly explanation.

We use TreeExplainer, which is exact and fast on tree models like LightGBM.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import shap

from src.data import preprocessor as pp
from src.ml.predict import load_artifact
from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _get_explainer():
    """Build and cache a SHAP TreeExplainer for the saved model."""
    art = load_artifact()
    explainer = shap.TreeExplainer(art["model"])
    return explainer, art["preprocess_config"]


def _humanize(feature: str, value, shap_val: float) -> str:
    """Turn a (feature, value, shap) triple into a plain-English phrase."""
    direction = "increased" if shap_val > 0 else "decreased"
    pretty = feature.replace("_", " ").title()
    try:
        val_str = f"{float(value):.3g}"
    except (TypeError, ValueError):
        val_str = str(value)
    return f"{pretty} = {val_str} {direction} risk"


def explain_one(applicant: dict, top_n: int = 8) -> dict:
    """Explain a single prediction.

    Returns top features pushing risk UP and DOWN, with signed SHAP values and
    human-readable phrases.
    """
    explainer, pc = _get_explainer()
    df = pd.DataFrame([applicant])
    X = pp.transform(df, pc)

    shap_values = explainer.shap_values(X)
    # LightGBM binary: shap_values may be a list [class0, class1] or a 2D array.
    if isinstance(shap_values, list):
        vals = np.asarray(shap_values[1])[0]
    else:
        arr = np.asarray(shap_values)
        vals = arr[0]

    contributions = []
    for feat, sv in zip(pc.feature_names, vals):
        raw_val = applicant.get(feat, X.iloc[0][feat])
        contributions.append({
            "feature": feat,
            "value": (None if pd.isna(raw_val) else
                      (float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val))),
            "shap_value": float(sv),
            "explanation": _humanize(feat, raw_val, float(sv)),
        })

    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    increases = [c for c in contributions if c["shap_value"] > 0][:top_n]
    decreases = [c for c in contributions if c["shap_value"] < 0][:top_n]

    return {
        "top_risk_increasing": increases,
        "top_risk_decreasing": decreases,
        "base_value": float(np.ravel(explainer.expected_value)[-1]),
    }


def save_global_summary(max_display: int = 20, sample: int = 2000) -> str:
    """Save a global SHAP feature-importance bar plot for the UI/README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.loader import load_application_train

    explainer, pc = _get_explainer()
    df = load_application_train().sample(sample, random_state=42)
    X = pp.transform(df, pc)
    shap_values = explainer.shap_values(X)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(vals, X, max_display=max_display, plot_type="bar", show=False)
    out_path = config.MODELS_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    log.info("Saved SHAP summary -> %s", out_path)
    return str(out_path)


if __name__ == "__main__":
    import json
    sample = {
        "AMT_INCOME_TOTAL": 120000, "AMT_CREDIT": 600000, "AMT_ANNUITY": 25000,
        "DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -1500,
        "EXT_SOURCE_1": 0.2, "EXT_SOURCE_2": 0.3, "EXT_SOURCE_3": 0.25,
        "CODE_GENDER": "M", "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    }
    print(json.dumps(explain_one(sample), indent=2)[:1500])
