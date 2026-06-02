"""Data loading utilities.

Two jobs:
1. Read the Home Credit CSVs into pandas DataFrames (with memory-friendly dtype
   downcasting, because application_train.csv is ~159 MB / 122 columns).
2. Build a SQLite database holding the `applications` table, which the
   talk-to-data chatbot queries. Also emit sql/schema.sql for documentation.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric columns to the smallest safe dtype to save memory."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_application_train(downcast: bool = True) -> pd.DataFrame:
    """Load application_train.csv (the main table with the TARGET column)."""
    log.info("Loading %s", config.APPLICATION_TRAIN.name)
    df = pd.read_csv(config.APPLICATION_TRAIN)
    if downcast:
        df = _downcast(df)
    log.info("Loaded train: %d rows x %d cols", df.shape[0], df.shape[1])
    return df


def load_application_test(downcast: bool = True) -> pd.DataFrame:
    """Load application_test.csv (no TARGET; used for inference demos)."""
    df = pd.read_csv(config.APPLICATION_TEST)
    if downcast:
        df = _downcast(df)
    return df


def load_columns_description() -> pd.DataFrame:
    """Load the human-readable column descriptions (used in LLM prompts)."""
    # This file has a stray index column and latin-1 encoding in places.
    return pd.read_csv(config.COLUMNS_DESCRIPTION, encoding="latin-1")


def build_sqlite_db(db_path: Path | None = None, sample: int | None = None) -> Path:
    """Write application_train into a SQLite `applications` table.

    The chatbot runs read-only SQL against this DB. We keep it to the single
    main table to stay simple and predictable for NL->SQL.

    Args:
        db_path: output path (defaults to config.DB_PATH).
        sample: if given, only write this many rows (useful for quick tests).
    """
    db_path = Path(db_path) if db_path else config.DB_PATH
    config.ensure_dirs()

    df = load_application_train()
    if sample:
        df = df.head(sample)

    log.info("Writing SQLite DB -> %s", db_path)
    # Replace so re-runs are idempotent.
    with sqlite3.connect(db_path) as conn:
        df.to_sql(config.APPLICATIONS_TABLE, conn, if_exists="replace", index=False)
        # A couple of indexes that help common chatbot queries.
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_target "
            f"ON {config.APPLICATIONS_TABLE}(TARGET)"
        )

    _write_schema_sql(df)
    log.info("SQLite DB ready: %d rows in table '%s'", len(df), config.APPLICATIONS_TABLE)
    return db_path


def _write_schema_sql(df: pd.DataFrame) -> None:
    """Emit sql/schema.sql documenting the applications table columns."""
    type_map = {"int": "INTEGER", "float": "REAL", "object": "TEXT", "bool": "INTEGER"}
    lines = [f"CREATE TABLE {config.APPLICATIONS_TABLE} ("]
    col_defs = []
    for col, dtype in df.dtypes.items():
        kind = "TEXT"
        for key, sql_type in type_map.items():
            if key in str(dtype):
                kind = sql_type
                break
        col_defs.append(f"    {col} {kind}")
    lines.append(",\n".join(col_defs))
    lines.append(");")
    config.SCHEMA_SQL_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote schema -> %s", config.SCHEMA_SQL_PATH)


if __name__ == "__main__":
    build_sqlite_db()
