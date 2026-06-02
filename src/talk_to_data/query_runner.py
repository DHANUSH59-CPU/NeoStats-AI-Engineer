"""Validate and execute LLM-generated SQL safely against the SQLite DB.

This is the hallucination / safety-control layer. The LLM is untrusted: it may
produce dangerous (DROP), malformed, or off-schema SQL. Before anything touches
the database we:
  * strip markdown fences the model sometimes adds,
  * confirm it is a SINGLE statement,
  * confirm it STARTS WITH SELECT and contains no DML/DDL keywords,
  * confirm it only references the known table,
  * enforce a row LIMIT.
Execution uses a read-only connection as defence in depth.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import pandas as pd

from src.talk_to_data.prompt_templates import TABLE_NAME
from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


@dataclass
class SQLValidationError(Exception):
    """Raised when generated SQL fails a safety check."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def clean_sql(raw: str) -> str:
    """Strip markdown fences / 'sql:' prefixes the model may add."""
    text = raw.strip()
    text = re.sub(r"^```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^sql\s*:?", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(";").strip()


def validate_sql(sql: str) -> str:
    """Run all safety checks. Returns the (possibly LIMIT-augmented) SQL or raises."""
    sql = clean_sql(sql)

    if not sql:
        raise SQLValidationError("Empty query.")

    # Single statement only (no stacked queries).
    if ";" in sql:
        raise SQLValidationError("Multiple statements are not allowed.")

    # Must be a SELECT (allow a leading WITH ... SELECT CTE).
    lowered = sql.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise SQLValidationError("Only SELECT queries are allowed.")

    # No DML / DDL keywords anywhere.
    if _FORBIDDEN.search(sql):
        raise SQLValidationError("Query contains a forbidden keyword (write/DDL).")

    # Only the known table may be referenced.
    tables = set(re.findall(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE))
    unknown = tables - {TABLE_NAME}
    if unknown:
        raise SQLValidationError(f"Unknown table(s): {', '.join(sorted(unknown))}.")

    # Enforce a row cap.
    if "limit" not in lowered:
        sql = f"{sql} LIMIT {config.SQL_ROW_LIMIT}"

    return sql


def run_sql(sql: str) -> pd.DataFrame:
    """Execute validated SQL read-only and return a DataFrame."""
    # Read-only URI connection: even if validation were bypassed, writes fail.
    uri = f"file:{config.DB_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return pd.read_sql_query(sql, conn)


def validate_and_run(sql: str) -> tuple[str, pd.DataFrame]:
    """Validate then execute. Returns (final_sql, result_df)."""
    final_sql = validate_sql(sql)
    log.info("Running SQL: %s", final_sql)
    return final_sql, run_sql(final_sql)


if __name__ == "__main__":
    # Demonstrate: a good query runs; a malicious one is blocked.
    ok_sql = "SELECT ROUND(AVG(TARGET)*100,2) AS default_rate_pct FROM applications"
    final, df = validate_and_run(ok_sql)
    print("OK ->", df.to_dict("records"))

    for bad in ["DROP TABLE applications",
                "SELECT * FROM applications; DROP TABLE applications",
                "SELECT * FROM secret_table"]:
        try:
            validate_sql(bad)
            print("FAILED TO BLOCK:", bad)
        except SQLValidationError as e:
            print("Blocked:", bad, "->", e)
