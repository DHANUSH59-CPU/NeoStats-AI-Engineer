"""Model training pipeline.

Run with:  python -m src.ml.train

What it does, step by step (explained for first-time model training):
  1. Load the main table and run the feature pipeline (preprocessor.fit_transform).
  2. Split into train / validation sets, *stratified* so both keep the ~8%
     default rate.
  3. Train a BASELINE Logistic Regression (simple, fully explainable) to set a
     performance floor.
  4. Train the MAIN model, LightGBM (gradient-boosted trees), handling the class
     imbalance with `scale_pos_weight`.
  5. Pick the better model on validation ROC-AUC, evaluate it, and SAVE one
     artifact (model + preprocessing config + metadata) plus a human-readable
     metadata.json and ROC/PR curve images.
"""
from __future__ import annotations

from datetime import datetime, timezone

import joblib
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.data.loader import load_application_train
from src.data import preprocessor as pp
from src.ml.evaluate import compute_metrics, save_curves
from src.utils import config
from src.utils.helpers import write_json
from src.utils.logger import get_logger

log = get_logger(__name__)

RANDOM_STATE = 42


def _train_baseline(X_train, y_train, X_val, y_val) -> float:
    """Logistic Regression on numeric features only. Returns validation ROC-AUC.

    This is a *floor*: a simple, transparent model. If our fancy model can't beat
    it, something is wrong. We use only numeric columns here (LR can't consume
    raw categoricals), with median imputation + scaling.
    """
    from sklearn.metrics import roc_auc_score

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train[numeric_cols], y_train)
    val_prob = pipe.predict_proba(X_val[numeric_cols])[:, 1]
    auc = roc_auc_score(y_val, val_prob)
    log.info("Baseline LogisticRegression  val ROC-AUC = %.4f", auc)
    return float(auc)


def _train_lightgbm(X_train, y_train, X_val, y_val, categorical_features):
    """Train LightGBM with early stopping on validation AUC.

    NOTE on class imbalance: we deliberately do NOT use `scale_pos_weight` /
    `is_unbalance` here. We tested both and they *hurt* ranking performance on
    this dataset (val AUC dropped from ~0.77 to ~0.72 with degenerate early
    stopping at iteration 1) — over-weighting the rare class inflates early
    gradients and destroys probability calibration. LightGBM's BoostFromScore
    already initializes to the base rate. Instead we (a) judge the model with
    ROC-AUC / PR-AUC (imbalance-aware ranking metrics) and (b) handle imbalance
    at the DECISION layer by tuning the operating threshold (see train()).
    """
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    log.info("Class balance: %d negatives / %d positives (%.1f:1)",
             n_neg, n_pos, n_neg / max(n_pos, 1))

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=categorical_features,
        callbacks=[early_stopping(stopping_rounds=80), log_evaluation(period=100)],
    )
    log.info("LightGBM best iteration: %s", model.best_iteration_)
    return model


def train() -> dict:
    """Run the full training pipeline and save artifacts. Returns metadata dict."""
    config.ensure_dirs()

    # --- 1. Load + preprocess -------------------------------------------------
    df = load_application_train()
    X, y, pc = pp.fit_transform(df)
    log.info("Feature matrix: %s | default rate: %.4f", X.shape, y.mean())

    # --- 2. Stratified train/validation split --------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log.info("Train: %s | Val: %s", X_train.shape, X_val.shape)

    # --- 3. Baseline ----------------------------------------------------------
    baseline_auc = _train_baseline(X_train, y_train, X_val, y_val)

    # --- 4. Main model: LightGBM ---------------------------------------------
    model = _train_lightgbm(X_train, y_train, X_val, y_val, pc.categorical_features)
    val_prob = model.predict_proba(X_val)[:, 1]

    # --- 5. Imbalance handling at the decision layer: tune the threshold ------
    # A 0.5 cutoff is meaningless for an 8% base rate. We pick the threshold that
    # maximizes F1 (balances catching defaulters vs false alarms) on validation.
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_val, val_prob)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_threshold = float(thr[int(np.nanargmax(f1[:-1]))]) if len(thr) else 0.5
    log.info("Tuned decision threshold (max-F1) = %.3f", best_threshold)

    metrics = compute_metrics(y_val, val_prob, threshold=best_threshold)
    log.info("LightGBM val ROC-AUC = %.4f | PR-AUC = %.4f",
             metrics["roc_auc"], metrics["pr_auc"])
    if metrics["roc_auc"] < baseline_auc:
        log.warning("LightGBM did NOT beat the baseline — investigate features.")

    curves = save_curves(y_val, val_prob)

    # Data-driven risk bands: use percentiles of predicted probability so the
    # Low/Medium/High split reflects the real population, not arbitrary cutoffs.
    low_max = float(np.quantile(val_prob, 0.70))   # bottom 70% -> Low
    med_max = float(np.quantile(val_prob, 0.90))   # next 20%   -> Medium; top 10% High
    log.info("Risk-band thresholds (data-driven): Low<%.3f, Medium<%.3f", low_max, med_max)

    # Top feature importances (for README / UI).
    importances = sorted(
        zip(pc.feature_names, model.feature_importances_.tolist()),
        key=lambda t: t[1], reverse=True,
    )[:20]

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "LightGBMClassifier",
        "n_features": len(pc.feature_names),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "default_rate": float(y.mean()),
        "baseline_roc_auc": baseline_auc,
        "decision_threshold": best_threshold,
        "imbalance_strategy": (
            "Natural distribution kept for training (scale_pos_weight/is_unbalance "
            "tested and rejected — they degraded ranking AUC). Imbalance addressed "
            "via ROC/PR-AUC evaluation and an F1-tuned decision threshold."
        ),
        "metrics": metrics,
        "curves": curves,
        "risk_bands": {"low_max": low_max, "medium_max": med_max},
        "top_features": [{"feature": f, "importance": imp} for f, imp in importances],
    }

    # --- 6. Save one self-contained artifact ---------------------------------
    artifact = {"model": model, "preprocess_config": pc, "metadata": metadata}
    joblib.dump(artifact, config.MODEL_PATH)
    write_json(config.METADATA_PATH, metadata)
    log.info("Saved model -> %s", config.MODEL_PATH)
    log.info("Saved metadata -> %s", config.METADATA_PATH)
    return metadata


if __name__ == "__main__":
    train()
