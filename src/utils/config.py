"""Central configuration for the Credit Risk platform.

Everything that might change between environments (file paths, the LLM model,
risk-band thresholds) lives here so the rest of the code never hard-codes it.
Values come from environment variables (loaded from a local `.env` if present),
with sensible defaults so the project also runs with zero configuration.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if it exists (no-op in production if absent).
load_dotenv()

# --------------------------------------------------------------------------
# Paths. PROJECT_ROOT is two levels up from this file: src/utils/config.py.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env_var: str, default: str) -> Path:
    """Resolve a path from an env var, relative to PROJECT_ROOT if not absolute."""
    raw = os.getenv(env_var, default)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


DATA_DIR: Path = _path("DATA_DIR", "data/home-credit-default-risk")
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = _path("MODEL_PATH", "models/model.pkl")
DB_PATH: Path = _path("DB_PATH", "models/credit.db")

# Key dataset files.
APPLICATION_TRAIN = DATA_DIR / "application_train.csv"
APPLICATION_TEST = DATA_DIR / "application_test.csv"
BUREAU = DATA_DIR / "bureau.csv"
PREVIOUS_APPLICATION = DATA_DIR / "previous_application.csv"
COLUMNS_DESCRIPTION = DATA_DIR / "HomeCredit_columns_description.csv"

# Derived artifacts written by the pipeline.
EDA_SUMMARY_PATH = MODELS_DIR / "eda_summary.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
RULES_PATH = MODELS_DIR / "rules.json"
SCHEMA_SQL_PATH = PROJECT_ROOT / "sql" / "schema.sql"

# --------------------------------------------------------------------------
# Database table name used by the talk-to-data chatbot.
# --------------------------------------------------------------------------
APPLICATIONS_TABLE = "applications"

# --------------------------------------------------------------------------
# LLM settings.
# --------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# --------------------------------------------------------------------------
# API server.
# --------------------------------------------------------------------------
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# --------------------------------------------------------------------------
# Talk-to-data safety.
# --------------------------------------------------------------------------
SQL_ROW_LIMIT = int(os.getenv("SQL_ROW_LIMIT", "100"))

# --------------------------------------------------------------------------
# Risk bands. The model outputs a default probability in [0, 1]; we bucket it
# into business-readable bands. Thresholds are revisited in train.py against the
# real score distribution, but these defaults are reasonable for ~8% base rate.
# --------------------------------------------------------------------------
RISK_BAND_LOW_MAX = 0.10      # prob < 0.10           -> "Low"
RISK_BAND_MEDIUM_MAX = 0.30   # 0.10 <= prob < 0.30   -> "Medium"
#                               prob >= 0.30           -> "High"


def risk_band(probability: float) -> str:
    """Map a default probability to a Low / Medium / High risk band."""
    if probability < RISK_BAND_LOW_MAX:
        return "Low"
    if probability < RISK_BAND_MEDIUM_MAX:
        return "Medium"
    return "High"


# Map a risk band to an underwriting action — turns the score into a decision
# a credit officer can act on (decision-support, not auto-decisioning).
APPROVAL_BY_BAND = {"Low": "Approve", "Medium": "Manual Review", "High": "Decline"}


def approval_recommendation(band: str) -> str:
    """Risk band -> underwriting recommendation (Approve / Manual Review / Decline)."""
    return APPROVAL_BY_BAND.get(band, "Manual Review")


def ensure_dirs() -> None:
    """Create output directories that the pipeline writes to."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_SQL_PATH.parent.mkdir(parents=True, exist_ok=True)
