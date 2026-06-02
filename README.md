# 🏦 AI-Powered Credit Risk Intelligence Platform

An end-to-end, explainable credit-risk platform built on the [Home Credit Default
Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset.
It scores loan applicants, explains every decision, derives business rules, and
lets analysts query the data in plain English — all behind one FastAPI app you
launch with a single `docker-compose up`.

> NeoStats AI Engineer assignment submission.

---

## ✨ What it does (5 modules)

| Module | Description |
|---|---|
| **📊 EDA** | Dataset summary, data-quality findings, 5+ business insights with charts. |
| **🎯 Risk Prediction** | LightGBM model → default probability + Low/Medium/High risk band + live metrics. |
| **🔍 Explainability** | SHAP shows the exact features that pushed each applicant's risk up/down. |
| **📋 Decision Rules** | Business-readable IF/THEN rules (with support & default rate) bridging ML and credit policy. |
| **💬 Talk-to-Data** | Natural-language → SQL chatbot (Google Gemini) with a strict SQL-safety layer. |

---

## 🏗️ Architecture

```
                          ┌──────────────────────────────┐
   Browser (5 tabs)  ───▶ │   FastAPI app (src/api)      │
   frontend/ static       │   serves frontend + /api/*   │
                          └───────────────┬──────────────┘
            ┌──────────────┬──────────────┼──────────────┬───────────────┐
            ▼              ▼              ▼              ▼               ▼
        /api/eda      /api/predict   /api/explain    /api/rules      /api/chat
            │              │              │              │               │
     eda_summary.json  ml/predict   explain/shap    rules.json   talk_to_data/
     (notebooks/eda)   (model.pkl)  (TreeExplainer)              nl_to_sql→Gemini
                           │                                      query_runner
                    data/preprocessor                            (SQL validation)
                    (shared fit/transform)                         credit.db (SQLite)
```

- **Single container**: FastAPI serves both the JSON API and the static frontend.
- **Shared preprocessing**: `src/data/preprocessor.py` runs identically at train
  and predict time (no training/serving skew).
- **Artifacts** live in `models/` and are built on first run by `src/bootstrap.py`.

---

## 📁 Project structure

```
credit_risk_platform/
├── data/home-credit-default-risk/   # dataset (mounted, not committed)
├── documents/project_presentation.pdf
├── notebooks/eda.ipynb, eda.py      # EDA (notebook + script)
├── src/
│   ├── data/        loader.py, preprocessor.py
│   ├── ml/          train.py, predict.py, evaluate.py
│   ├── explain/     shap_explainer.py
│   ├── rules/       derive_rules.py
│   ├── talk_to_data/ nl_to_sql.py, query_runner.py, prompt_templates.py
│   ├── api/         main.py, routers/, schemas.py
│   ├── utils/       config.py, logger.py, helpers.py
│   └── bootstrap.py  # builds missing artifacts on container start
├── frontend/        index.html, app.js, style.css
├── sql/schema.sql
├── models/          model.pkl, metadata.json, *.json, *.png, credit.db
├── Dockerfile, docker-compose.yml, .env.example, requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

| For | You need |
|---|---|
| Docker run (Option A) | Docker Desktop / Docker Engine + Compose |
| Local run (Option B) | Python 3.10 |
| Both | A Google **Gemini API key** (free: https://aistudio.google.com/apikey) — *optional*, the chatbot falls back to offline queries without one |

### 📥 Required: download the dataset (≈2.6 GB)

The dataset is **not** included in this repo (too large for git). Before running,
download it and place the CSVs in `data/home-credit-default-risk/`.

1. Go to the Kaggle competition data page:
   **https://www.kaggle.com/competitions/home-credit-default-risk/data**
   (sign in and accept the competition rules to enable download).
2. Download and unzip so the folder looks like this:
   ```
   data/home-credit-default-risk/
   ├── application_train.csv          ← REQUIRED (the app, DB & model use this)
   ├── application_test.csv
   ├── HomeCredit_columns_description.csv
   └── (bureau.csv, previous_application.csv, … — optional, not needed to run)
   ```
   > **Minimum to run:** only `application_train.csv` is required. The other tables
   > aren't used by the shipped single-table model/chatbot.

   *Tip (Kaggle CLI):* `kaggle competitions download -c home-credit-default-risk -p data/home-credit-default-risk && unzip -o data/home-credit-default-risk/*.zip -d data/home-credit-default-risk`

If `data/home-credit-default-risk/application_train.csv` is missing, the container
starts but the app has no data — so don't skip this step.

---

## 🚀 Quick start

### Option A — Docker (recommended)

1. **Download the dataset** into `data/home-credit-default-risk/` — see
   [Prerequisites](#️-prerequisites) above. (At minimum `application_train.csv`.)
2. **Configure env**:
   ```bash
   cp .env.example .env
   # edit .env and set GOOGLE_API_KEY (free key: https://aistudio.google.com/apikey)
   ```
3. **Run**:
   ```bash
   docker-compose up --build
   ```
4. Open **http://localhost:8000**.

On first start the container builds the SQLite DB, EDA summary, and rules, and
trains the model if `models/model.pkl` is absent (a pre-trained model is shipped,
so this is usually skipped). Subsequent starts are fast.

### Option B — Local (Python 3.10)

```bash
pip install -r requirements.txt
cp .env.example .env                      # set GOOGLE_API_KEY
python -m src.data.loader                 # build SQLite DB
python -m src.ml.train                    # train + save model
python notebooks/eda.py                   # EDA summary
python -m src.rules.derive_rules          # derive rules
python -m uvicorn src.api.main:app --port 8000
```

> **No API key (or rate-limited)?** The chatbot falls back to a built-in set of
> validated queries for the example questions, so the app always demos. With a
> key it answers arbitrary questions via Gemini. The Gemini **free tier is
> limited (~5 req/min, ~20/day per model)**; on a `429` the app transparently
> degrades to the offline generator + a deterministic summary — it never errors.

---

## 🤖 Model: selection rationale & class-imbalance strategy

**Model choice — LightGBM.** Gradient-boosted trees are the standard strong
baseline on this dataset: they handle the 122 mixed numeric/categorical features
and heavy missingness natively (categoricals passed as `category` dtype, no
fragile one-hot), train fast, and pair perfectly with SHAP's exact `TreeExplainer`.
A **Logistic Regression baseline** (`class_weight="balanced"`) sets a transparent
performance floor.

**Class imbalance (only ~8% defaults) — what we found.** We *tested*
`scale_pos_weight` and `is_unbalance=True`, the usual levers. **Both hurt**: they
inflate early gradients, cause degenerate early-stopping at iteration 1, and drop
validation ROC-AUC from **0.77 → 0.72**. So our strategy is:

1. **Keep the natural distribution** for training (LightGBM's `BoostFromScore`
   already initializes to the base rate, preserving calibration).
2. **Judge with imbalance-aware ranking metrics** — ROC-AUC and **PR-AUC** —
   never plain accuracy (which is a misleading 92% for an all-"no-default" model).
3. **Handle imbalance at the decision layer**: tune the operating threshold to
   **maximize F1** instead of using a meaningless 0.5 cutoff.

This is documented in `metadata.json["imbalance_strategy"]`.

### 📈 Evaluation metrics (held-out 20% validation, 61,503 rows)

| Metric | LightGBM | Baseline (LogReg) |
|---|---|---|
| **ROC-AUC** | **0.770** | 0.738 |
| **PR-AUC** | **0.259** | — |
| **KS statistic** | **0.407** | — |
| Decision threshold (max-F1) | 0.148 | — |
| Precision / Recall @ threshold | 0.25 / 0.44 | — |

ROC and PR curves are saved to `models/roc_curve.png` and `models/pr_curve.png`.
The **KS statistic** (Kolmogorov–Smirnov) is the standard credit-scoring
separation metric — the maximum gap between the cumulative true- and
false-positive rates; 0.40+ indicates solid discrimination.

**Risk bands** (data-driven from the score distribution): Low < 0.084,
Medium < 0.182, High ≥ 0.182 — each mapped to an **underwriting recommendation**:
Low → **Approve**, Medium → **Manual Review**, High → **Decline** (decision-support,
surfaced in the prediction UI alongside the score).

**Top drivers** (gain importance): `ORGANIZATION_TYPE`, `CREDIT_TERM`,
`EXT_SOURCE_3/1/2`, then `DAYS_*` and the engineered ratios
(`CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `EMPLOYED_AGE_RATIO`).

---

## 💬 Talk-to-Data: prompt engineering & hallucination control

**Prompt engineering & token optimization** (`src/talk_to_data/prompt_templates.py`):
- The prompt includes a **curated 24-column schema**, not all 122 columns —
  smaller prompt = cheaper, faster, fewer hallucinations.
- The schema string is built **once at import** (cached) and reused per request.
- **Few-shot examples** teach the SQLite dialect and the metric conventions
  (e.g. "default rate = `AVG(TARGET)*100`"). `temperature=0` for deterministic SQL.
- The model is instructed to return **SQL only**; an unanswerable question must
  return a sentinel (`SELECT 'UNSUPPORTED' …`).

**Hallucination / safety control** (`src/talk_to_data/query_runner.py`) — the LLM
is untrusted, so every generated query is validated before execution:
- strips markdown fences,
- must be a **single statement** (no stacked `;` queries),
- must **start with `SELECT`/`WITH`** and contain **no** `INSERT/UPDATE/DELETE/DROP/…`,
- may reference **only** the `applications` table,
- a **row `LIMIT`** is enforced,
- execution uses a **read-only** SQLite connection (defence in depth).

The UI shows the **exact SQL and raw rows** alongside the prose answer, so every
result is auditable. 6 demo questions ship in the chatbot.

---

## 📋 Rule derivation logic & sample outputs

`src/rules/derive_rules.py` fits a **shallow decision tree** (depth 4, min 2000
samples/leaf) on interpretable features (external scores + engineered ratios).
Each root→leaf path becomes a rule; we report **support**, **observed default
rate**, **lift vs base rate (8.1%)**, and a Low/Medium/High label.

Sample (from `models/rules.json`):

| Risk | Rule | Default rate | Applicants | Lift |
|---|---|---|---|---|
| High | `EXT_SOURCE_3 ≤ 0.315 AND EXT_SOURCE_2 ≤ 0.080` | 37.6% | 2,079 | 4.66× |
| High | `EXT_SOURCE_3 ≤ 0.315 AND EXT_SOURCE_2 ≤ 0.380 AND EXT_SOURCE_3 ≤ 0.146` | 31.4% | 2,845 | 3.89× |
| High | `EXT_SOURCE_3 > 0.315 AND EXT_SOURCE_2 ≤ 0.153 …` | 21.7% | 9,024 | 2.69× |

---

## 🧠 EDA — key findings

1. Only **8.07%** of applicants default → heavy class imbalance.
2. **Lower education** defaults ~10.9% vs ~5% for higher education.
3. **Younger applicants (20–30)** default markedly more than 50–60+.
4. **External credit scores** (`EXT_SOURCE_*`) are the strongest signals.
5. **Low-skill Laborers / Drivers** have the highest occupational default rates (>13%).
6. `DAYS_EMPLOYED` has an **18% sentinel anomaly** (`365243` = pensioners), handled in preprocessing.

Full analysis: `notebooks/eda.ipynb`. Regenerate with `python notebooks/eda.py`.

---

## 🔌 API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness + model presence |
| GET | `/api/eda` | EDA summary & chart data |
| POST | `/api/predict` | Applicant → probability + risk band + metrics |
| POST | `/api/explain` | Applicant → SHAP risk drivers |
| GET | `/api/rules` | Derived business rules |
| POST | `/api/chat` | NL question → `{sql, answer, rows}` |
| GET | `/api/chat/examples` | Suggested questions |

Interactive docs at `/docs` (FastAPI Swagger UI).

---

## 🔧 Tech stack & rationale

- **ML**: scikit-learn + **LightGBM** + **SHAP**.
- **Backend**: **FastAPI** + uvicorn (async, auto-docs, easy to Dockerize).
- **Frontend**: static HTML + vanilla JS + **Chart.js** — no build step, one
  container. (We chose FastAPI + a lightweight frontend over Streamlit for a
  cleaner API/UI separation.)
- **LLM**: **Google Gemini** (`gemini-flash-latest`) — free tier, with automatic
  offline fallback when rate-limited.
- **DB**: **SQLite** — zero-config, file-based, ships in the container.

---

## ⚠️ Known limitations & improvements

- **Single-table model.** Only `application_train.csv` is used. Aggregating
  `bureau.csv` / `previous_application.csv` (counts, overdue sums, approval rates)
  typically lifts AUC to ~0.78–0.79 — a clear next step (the preprocessor is
  structured to add these).
- **No hyperparameter search.** Sensible fixed LightGBM params; Optuna tuning
  would help marginally.
- **Chatbot scope.** Restricted to the single `applications` table by design
  (safety). Multi-table NL→SQL would need schema-linking.
- **Probability calibration.** Bands are percentile-based; Platt/isotonic
  calibration would make probabilities more literally interpretable.
- **Frontend uses a Chart.js CDN** — needs internet on the evaluator's machine.

---

## 🗂️ Major design decisions (brief)

1. **Shared preprocessing functions** persisted inside the model artifact →
   zero training/serving skew.
2. **Rejected class reweighting** after measuring that it hurt ranking — imbalance
   handled via metrics + threshold instead. (Honest, data-driven.)
3. **SQL safety as a separate, testable layer** independent of the LLM.
4. **Bootstrap-on-start** so `docker-compose up` is truly one command.
