"""Inference: score a single applicant.

Loads the saved artifact once (model + preprocessing config + metadata) and
exposes `predict_one(applicant_dict)` returning the default probability, the
risk band, and the model's headline metrics (so the UI can show them).
"""
from __future__ import annotations

import functools

import joblib
import pandas as pd

from src.data import preprocessor as pp
from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def load_artifact() -> dict:
    """Load and cache the model artifact from disk."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. Run `python -m src.ml.train` first."
        )
    log.info("Loading model artifact: %s", config.MODEL_PATH)
    return joblib.load(config.MODEL_PATH)


def _risk_band(prob: float, bands: dict | None) -> str:
    """Map probability -> band using data-driven thresholds if available."""
    if bands:
        if prob < bands["low_max"]:
            return "Low"
        if prob < bands["medium_max"]:
            return "Medium"
        return "High"
    return config.risk_band(prob)


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of raw applicant rows. Returns probability + band."""
    art = load_artifact()
    model = art["model"]
    pc = art["preprocess_config"]
    bands = art["metadata"].get("risk_bands")

    X = pp.transform(df, pc)
    probs = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"default_probability": probs})
    out["risk_band"] = [_risk_band(float(p), bands) for p in probs]
    return out


def predict_one(applicant: dict) -> dict:
    """Score a single applicant given as a dict of raw column -> value.

    Missing columns are fine: the preprocessor imputes them with training
    medians / treats categoricals as unknown.

    Returns:
        {
          "default_probability": float,
          "risk_band": "Low" | "Medium" | "High",
          "model_metrics": {... roc_auc, pr_auc ...},
        }
    """
    df = pd.DataFrame([applicant])
    result = predict_df(df).iloc[0]
    art = load_artifact()
    band = str(result["risk_band"])
    m = art["metadata"]["metrics"]
    return {
        "default_probability": round(float(result["default_probability"]), 4),
        "risk_band": band,
        "recommendation": config.approval_recommendation(band),
        "model_metrics": {
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "ks_statistic": m.get("ks_statistic"),
        },
    }


if __name__ == "__main__":
    # Tiny smoke test using a couple of representative fields.
    sample = {
        "AMT_INCOME_TOTAL": 120000,
        "AMT_CREDIT": 600000,
        "AMT_ANNUITY": 25000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -1500,
        "EXT_SOURCE_1": 0.2,
        "EXT_SOURCE_2": 0.3,
        "EXT_SOURCE_3": 0.25,
        "CODE_GENDER": "M",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_CONTRACT_TYPE": "Cash loans",
    }
    import json
    print(json.dumps(predict_one(sample), indent=2))
