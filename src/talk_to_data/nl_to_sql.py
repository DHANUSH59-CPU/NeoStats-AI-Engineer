"""Natural-language -> SQL -> readable answer, using Google Gemini.

Pipeline:
  question --(Gemini + schema prompt)--> SQL
          --(query_runner: validate + execute)--> result rows
          --(Gemini answer prompt)--> plain-English answer

If no GOOGLE_API_KEY is configured we fall back to a tiny rule-based generator
for the canned example questions, so the app still demos without a key.
"""
from __future__ import annotations

import functools
import json

import pandas as pd

from src.talk_to_data import prompt_templates as pt
from src.talk_to_data.query_runner import SQLValidationError, validate_and_run
from src.utils import config
from src.utils.logger import get_logger

log = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _get_llm():
    """Construct the Gemini chat model (cached). Returns None if no API key."""
    if not config.GOOGLE_API_KEY:
        log.warning("GOOGLE_API_KEY not set — using offline fallback for SQL generation.")
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.0,   # deterministic SQL
        max_retries=1,     # fail fast on 429s instead of a long backoff storm
    )


# Minimal offline fallback: maps the canned demo questions to known-good SQL so
# the chatbot works in a no-key demo. The LLM path handles everything else.
_FALLBACK_SQL = {
    "overall default rate": "SELECT ROUND(AVG(TARGET)*100,2) AS default_rate_pct FROM applications",
    "education": "SELECT NAME_EDUCATION_TYPE, ROUND(AVG(TARGET)*100,2) AS default_rate_pct, COUNT(*) AS n FROM applications GROUP BY NAME_EDUCATION_TYPE ORDER BY default_rate_pct DESC",
    "income of defaulters": "SELECT TARGET, ROUND(AVG(AMT_INCOME_TOTAL),0) AS avg_income FROM applications GROUP BY TARGET",
    "occupation": "SELECT OCCUPATION_TYPE, ROUND(AVG(TARGET)*100,2) AS default_rate_pct, COUNT(*) AS n FROM applications GROUP BY OCCUPATION_TYPE HAVING n>=1000 ORDER BY default_rate_pct DESC LIMIT 5",
    "own a car": "SELECT FLAG_OWN_CAR, COUNT(*) AS n, ROUND(AVG(TARGET)*100,2) AS default_rate_pct FROM applications GROUP BY FLAG_OWN_CAR",
    "cash loans versus revolving": "SELECT NAME_CONTRACT_TYPE, ROUND(AVG(TARGET)*100,2) AS default_rate_pct, COUNT(*) AS n FROM applications GROUP BY NAME_CONTRACT_TYPE",
}


def _fallback_sql(question: str) -> str:
    q = question.lower()
    for key, sql in _FALLBACK_SQL.items():
        if all(word in q for word in key.split()):
            return sql
    # default: overall default rate
    return _FALLBACK_SQL["overall default rate"]


def generate_sql(question: str) -> tuple[str, str]:
    """Turn a question into SQL. Returns (sql, source) where source is 'gemini'
    or 'fallback'.

    If NO API key is configured we use the offline generator (so the app still
    demos). If a key IS configured and Gemini errors (e.g. a 429 rate-limit),
    we let the error propagate so the chatbot surfaces it to the user.
    """
    llm = _get_llm()
    if llm is None:
        return _fallback_sql(question), "fallback"

    messages = [
        ("system", pt.SYSTEM_PROMPT),
        ("human", question),
    ]
    resp = llm.invoke(messages)  # errors propagate to ask() -> shown in UI
    sql = resp.content if hasattr(resp, "content") else str(resp)
    return sql, "gemini"


def _deterministic_summary(df: pd.DataFrame, rows: list[dict]) -> str:
    """A readable answer built without the LLM (used as a fallback)."""
    if df.empty:
        return "The query returned no rows."
    if len(df) == 1 and len(df.columns) == 1:
        col = df.columns[0]
        return f"{col.replace('_', ' ')}: {rows[0][col]}"
    return f"Query returned {len(df)} row(s). Top result: {rows[0]}"


def summarize_answer(question: str, sql: str, df: pd.DataFrame) -> str:
    """Turn result rows into a plain-English answer.

    No key (or empty result) -> deterministic summary. With a key, any Gemini
    error propagates so the chatbot can surface it to the user.
    """
    rows = df.head(config.SQL_ROW_LIMIT).to_dict("records")
    llm = _get_llm()
    if llm is None or df.empty:
        return _deterministic_summary(df, rows)

    messages = [
        ("system", pt.ANSWER_SYSTEM_PROMPT),
        ("human", f"Question: {question}\nSQL: {sql}\nResults (JSON): {json.dumps(rows, default=str)}"),
    ]
    resp = llm.invoke(messages)  # errors propagate to ask() -> shown in UI
    return resp.content if hasattr(resp, "content") else str(resp)


def ask(question: str) -> dict:
    """Full talk-to-data round trip. Returns sql, answer, rows, and any error.

    The response is fully auditable: the caller (and UI) sees the exact SQL that
    ran and the raw rows, not just the prose answer.
    """
    result = {"question": question, "sql": None, "answer": None,
              "rows": [], "columns": [], "error": None,
              "source": None, "prompt_version": pt.PROMPT_VERSION}
    try:
        raw_sql, source = generate_sql(question)
        result["source"] = source
        final_sql, df = validate_and_run(raw_sql)
        result["sql"] = final_sql
        result["columns"] = df.columns.tolist()
        result["rows"] = df.head(config.SQL_ROW_LIMIT).to_dict("records")
        result["answer"] = summarize_answer(question, final_sql, df)
    except SQLValidationError as e:
        result["error"] = f"Generated SQL was rejected by the safety check: {e}"
    except Exception as e:  # Gemini / other errors are surfaced to the UI
        log.exception("Talk-to-data failed")
        result["error"] = _format_llm_error(e)
    return result


def _format_llm_error(e: Exception) -> str:
    """Make a Gemini/API error readable for the chatbot (no giant gRPC dump)."""
    name = type(e).__name__
    msg = str(e)
    if "ResourceExhausted" in name or "429" in msg:
        return ("Gemini API error (429): free-tier quota exceeded "
                "(~5 requests/min, ~20/day). Please wait and retry.")
    if "NotFound" in name or "404" in msg:
        return (f"Gemini API error (404): model '{config.GEMINI_MODEL}' not found "
                "for this key. Set a valid GEMINI_MODEL in .env.")
    if "PermissionDenied" in name or "API_KEY" in msg or "401" in msg or "403" in msg:
        return "Gemini API error: invalid or unauthorized API key. Check GOOGLE_API_KEY in .env."
    # Fallback: show the first line of the real error.
    first_line = msg.strip().splitlines()[0] if msg.strip() else name
    return f"Gemini error: {first_line[:300]}"


if __name__ == "__main__":
    for q in pt.example_questions():
        out = ask(q)
        print(f"\nQ: {q}")
        print(f"SQL: {out['sql']}")
        print(f"A: {out['answer']}" if not out["error"] else f"ERROR: {out['error']}")
