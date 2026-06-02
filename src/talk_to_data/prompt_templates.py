"""Versioned prompt templates for the NL -> SQL talk-to-data agent.

Token-optimization strategy:
  * We do NOT dump all 122 columns into the prompt. We include a curated subset
    of ~25 business-relevant columns with one-line descriptions. This keeps the
    prompt small (cheaper, faster, fewer hallucinations) while covering the
    questions analysts actually ask.
  * The schema string is a module-level constant so it is built once and reused
    (effectively cached) across requests.
  * The model is told to return ONLY SQL (no prose), which we then validate.
"""
from __future__ import annotations

# Bump this when the prompt changes, so results are reproducible/traceable.
PROMPT_VERSION = "v1"

TABLE_NAME = "applications"

# Curated, high-value columns. (col_name, type, description)
SCHEMA_COLUMNS: list[tuple[str, str, str]] = [
    ("SK_ID_CURR", "INTEGER", "Unique loan/application ID"),
    ("TARGET", "INTEGER", "1 = client defaulted (payment difficulties), 0 = repaid"),
    ("NAME_CONTRACT_TYPE", "TEXT", "'Cash loans' or 'Revolving loans'"),
    ("CODE_GENDER", "TEXT", "Gender: 'M', 'F', or 'XNA'"),
    ("FLAG_OWN_CAR", "TEXT", "Owns a car: 'Y'/'N'"),
    ("FLAG_OWN_REALTY", "TEXT", "Owns property: 'Y'/'N'"),
    ("CNT_CHILDREN", "INTEGER", "Number of children"),
    ("AMT_INCOME_TOTAL", "REAL", "Annual income of the client"),
    ("AMT_CREDIT", "REAL", "Total credit amount of the loan"),
    ("AMT_ANNUITY", "REAL", "Loan annuity (yearly payment)"),
    ("AMT_GOODS_PRICE", "REAL", "Price of goods the loan is for"),
    ("NAME_INCOME_TYPE", "TEXT", "Income type e.g. 'Working','Pensioner','State servant'"),
    ("NAME_EDUCATION_TYPE", "TEXT", "Highest education level"),
    ("NAME_FAMILY_STATUS", "TEXT", "Family status e.g. 'Married','Single'"),
    ("NAME_HOUSING_TYPE", "TEXT", "Housing situation e.g. 'House / apartment'"),
    ("OCCUPATION_TYPE", "TEXT", "Occupation of the client (may be NULL)"),
    ("ORGANIZATION_TYPE", "TEXT", "Employer organization type"),
    ("DAYS_BIRTH", "INTEGER", "Age in days, NEGATIVE (e.g. -12000). Age years = -DAYS_BIRTH/365"),
    ("DAYS_EMPLOYED", "INTEGER", "Days employed before application, NEGATIVE. 365243 = unemployed"),
    ("CNT_FAM_MEMBERS", "REAL", "Number of family members"),
    ("EXT_SOURCE_1", "REAL", "Normalized external credit score 1 (0-1, higher=safer)"),
    ("EXT_SOURCE_2", "REAL", "Normalized external credit score 2 (0-1, higher=safer)"),
    ("EXT_SOURCE_3", "REAL", "Normalized external credit score 3 (0-1, higher=safer)"),
    ("REGION_RATING_CLIENT", "INTEGER", "Region rating 1/2/3 (3=worst)"),
]


def build_schema_string() -> str:
    """Return a compact schema description for the prompt."""
    lines = [f"Table `{TABLE_NAME}` (one row per loan application). Columns:"]
    for name, typ, desc in SCHEMA_COLUMNS:
        lines.append(f"  - {name} ({typ}): {desc}")
    return "\n".join(lines)


# Built once at import (cached).
SCHEMA_STRING = build_schema_string()

# Few-shot examples teach the model the dialect + tone of answer we want.
FEW_SHOT = """\
Example 1
Q: What is the overall default rate?
SQL: SELECT ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct FROM applications;

Example 2
Q: Default rate by education level, highest first.
SQL: SELECT NAME_EDUCATION_TYPE, ROUND(AVG(TARGET)*100, 2) AS default_rate_pct,
COUNT(*) AS n FROM applications GROUP BY NAME_EDUCATION_TYPE ORDER BY default_rate_pct DESC;

Example 3
Q: Average income of defaulters vs non-defaulters.
SQL: SELECT TARGET, ROUND(AVG(AMT_INCOME_TOTAL), 0) AS avg_income FROM applications GROUP BY TARGET;
"""

SYSTEM_PROMPT = f"""You are a careful SQL analyst for a bank's credit-risk database.
Convert the user's question into ONE valid SQLite SELECT query over this schema.

{SCHEMA_STRING}

STRICT RULES:
1. Output ONLY the SQL query. No explanation, no markdown fences, no comments.
2. Use a single SELECT statement. NEVER use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.
3. Use ONLY the table `{TABLE_NAME}` and ONLY columns listed above.
4. For "default rate" use AVG(TARGET); multiply by 100 for a percentage.
5. Ages: -DAYS_BIRTH/365. Remember DAYS_* are negative.
6. Round numeric aggregates for readability. Add LIMIT for "top N" questions.
7. If the question cannot be answered from these columns, return exactly: SELECT 'UNSUPPORTED' AS error;

{FEW_SHOT}
Now write the SQL for the user's question."""


# Template for turning query results back into a readable business answer.
ANSWER_SYSTEM_PROMPT = """You are a banking data analyst. Given a user's question, the
SQL that was run, and the result rows (as JSON), write ONE or TWO concise sentences
answering the question in plain business English. Use the actual numbers. Do not
invent data not present in the results."""


def example_questions() -> list[str]:
    """The 5+ demo questions used to verify the chatbot end to end."""
    return [
        "What is the overall default rate?",
        "Show the default rate by education level, highest first.",
        "What is the average income of defaulters versus non-defaulters?",
        "Which 5 occupation types have the highest default rate (min 1000 applicants)?",
        "How many applicants own a car, and what is their default rate?",
        "Compare default rate for cash loans versus revolving loans.",
    ]
