"""Feature preprocessing pipeline.

CRITICAL DESIGN POINT: the *exact same* transformations must run at training
time and at prediction time, otherwise the model sees different inputs than it
was trained on ("training/serving skew"). So all logic lives in pure functions
here, and train.py saves the resulting feature list + medians inside the model
artifact. predict.py then replays them.

The transforms are intentionally simple and explainable:
  * Fix known data anomalies (DAYS_EMPLOYED sentinel, sign of DAYS_* columns).
  * Add a handful of well-known, interpretable ratio features.
  * Leave categoricals as pandas `category` dtype so LightGBM handles them
    natively (no fragile one-hot encoding).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Columns that are identifiers / target, never used as model features.
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

# The DAYS_EMPLOYED column uses 365243 as a sentinel for "not employed".
DAYS_EMPLOYED_ANOMALY = 365243


@dataclass
class PreprocessConfig:
    """Everything predict-time needs to reproduce training-time preprocessing."""

    feature_names: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    # median value per numeric feature, for imputing missing inputs at predict time
    numeric_medians: dict[str, float] = field(default_factory=dict)
    # observed categories per categorical feature
    category_levels: dict[str, list[str]] = field(default_factory=dict)


def _fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Repair known data-quality issues in the raw application table."""
    df = df.copy()

    # 1. DAYS_EMPLOYED == 365243 means "not currently employed" -> treat as NaN.
    if "DAYS_EMPLOYED" in df:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)

    # 2. DAYS_* are negative (days before the application). Convert the two most
    #    intuitive ones to positive years for readability + engineered features.
    if "DAYS_BIRTH" in df:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).astype("float32")
    if "DAYS_EMPLOYED" in df:
        df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365.25).astype("float32")

    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few interpretable ratio features known to help on this dataset."""
    df = df.copy()
    eps = 1e-5  # avoid divide-by-zero

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        # how many annuity payments to repay the credit (loan "term" proxy)
        df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + eps)
    if {"YEARS_EMPLOYED", "AGE_YEARS"}.issubset(df.columns):
        df["EMPLOYED_AGE_RATIO"] = df["YEARS_EMPLOYED"] / (df["AGE_YEARS"] + eps)

    return df


def _raw_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Anomaly fixes + engineered features (shared by fit and transform)."""
    return _add_features(_fix_anomalies(df))


def fit_transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, PreprocessConfig]:
    """Training-time entry point.

    Returns:
        X: feature matrix (numeric + category-dtype columns)
        y: target series
        pc: PreprocessConfig to persist alongside the model
    """
    df = _raw_transform(df)

    y = df[TARGET_COL].astype("int8")
    drop_cols = [c for c in (ID_COL, TARGET_COL) if c in df.columns]
    X = df.drop(columns=drop_cols)

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    # Convert object columns to pandas 'category' dtype (LightGBM uses this).
    category_levels: dict[str, list[str]] = {}
    for col in categorical_features:
        X[col] = X[col].astype("category")
        category_levels[col] = X[col].cat.categories.astype(str).tolist()

    # Record medians so predict.py can impute missing numeric inputs.
    numeric_medians = {c: float(X[c].median()) for c in numeric_features}

    pc = PreprocessConfig(
        feature_names=X.columns.tolist(),
        categorical_features=categorical_features,
        numeric_medians=numeric_medians,
        category_levels=category_levels,
    )
    return X, y, pc


def transform(df: pd.DataFrame, pc: PreprocessConfig) -> pd.DataFrame:
    """Predict-time entry point. Reproduces fit_transform's feature space.

    Accepts a DataFrame of one or more raw applicant rows and returns a feature
    matrix with EXACTLY the columns/dtypes the model was trained on.
    """
    df = _raw_transform(df)

    # Build each trained feature column, then assemble in one shot (avoids the
    # DataFrame-fragmentation cost of inserting columns one at a time).
    cat_set = set(pc.categorical_features)
    cols: dict[str, pd.Series | pd.Categorical] = {}
    for col in pc.feature_names:
        if col in cat_set:
            series = df[col] if col in df.columns else pd.Series(index=df.index, dtype="object")
            # Pin to the categories seen in training; unseen values -> NaN.
            cols[col] = pd.Categorical(series.astype("object"),
                                       categories=pc.category_levels.get(col, []))
        else:
            series = df[col] if col in df.columns else pd.Series(index=df.index, dtype="float64")
            series = pd.to_numeric(series, errors="coerce")
            # Impute missing numerics with the training median.
            cols[col] = series.fillna(pc.numeric_medians.get(col, 0.0)).astype("float32")

    X = pd.DataFrame(cols, index=df.index)
    return X[pc.feature_names]
