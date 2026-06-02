"""Derive business-readable IF/THEN credit rules from the data.

Bridges ML and credit policy: we fit a *shallow* decision tree (depth 3-4) — a
"surrogate" that is intentionally simple enough that every path reads as a plain
rule. For each leaf we report the support (how many applicants land there) and
the observed default rate, then label the leaf Low/Medium/High risk.

Run with:  python -m src.rules.derive_rules
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, _tree

from src.data.loader import load_application_train
from src.data import preprocessor as pp
from src.utils import config
from src.utils.helpers import write_json
from src.utils.logger import get_logger

log = get_logger(__name__)

# We derive rules from interpretable numeric features only, so each rule reads
# cleanly (no one-hot dummy thresholds). These are strong, business-meaningful
# drivers on this dataset.
RULE_FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
    "AGE_YEARS", "YEARS_EMPLOYED", "AMT_CREDIT", "AMT_INCOME_TOTAL",
]


def _band_for_rate(rate: float, base_rate: float) -> str:
    """Label a leaf by how its default rate compares to the population base."""
    if rate >= 2.0 * base_rate:
        return "High"
    if rate >= 1.2 * base_rate:
        return "Medium"
    return "Low"


def _extract_rules(tree: DecisionTreeClassifier, feature_names, base_rate: float):
    """Walk the fitted tree and turn each root->leaf path into a rule dict."""
    t = tree.tree_
    rules = []

    def recurse(node, conditions):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[t.feature[node]]
            thr = t.threshold[node]
            recurse(t.children_left[node], conditions + [f"{name} <= {thr:.3f}"])
            recurse(t.children_right[node], conditions + [f"{name} > {thr:.3f}"])
        else:
            # Leaf. NOTE: sklearn >=1.4 stores `value` as class *fractions*
            # (summing to 1), not raw counts — so the default rate is value[1]
            # directly, and the true support is n_node_samples.
            fractions = t.value[node][0]
            support = int(t.n_node_samples[node])
            default_rate = float(fractions[1])
            rules.append({
                "conditions": conditions,
                "rule": " AND ".join(conditions) if conditions else "ALL applicants",
                "support": support,
                "default_rate": round(default_rate, 4),
                "risk_band": _band_for_rate(default_rate, base_rate),
                "lift": round(default_rate / base_rate, 2) if base_rate else None,
            })

    recurse(0, [])
    return rules


def derive_rules(max_depth: int = 4, min_samples_leaf: int = 2000) -> dict:
    """Fit the surrogate tree and write rules.json. Returns the rules payload."""
    config.ensure_dirs()
    df = load_application_train()
    df_feat = pp._raw_transform(df)  # noqa: SLF001 — reuse the shared transforms

    y = df_feat[pp.TARGET_COL].astype(int)
    base_rate = float(y.mean())

    available = [f for f in RULE_FEATURES if f in df_feat.columns]
    X = df_feat[available]
    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=available, index=X.index,
    )

    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42,
    )
    tree.fit(X_imp, y)

    rules = _extract_rules(tree, available, base_rate)
    # Most decision-relevant rules first: biggest deviation from base rate.
    rules.sort(key=lambda r: abs(r["default_rate"] - base_rate), reverse=True)

    payload = {
        "base_default_rate": round(base_rate, 4),
        "features_used": available,
        "n_rules": len(rules),
        "rules": rules,
    }
    write_json(config.RULES_PATH, payload)
    log.info("Derived %d rules (base rate %.3f) -> %s",
             len(rules), base_rate, config.RULES_PATH)
    for r in rules[:5]:
        log.info("  [%s] %s | rate=%.3f n=%d", r["risk_band"], r["rule"],
                 r["default_rate"], r["support"])
    return payload


if __name__ == "__main__":
    derive_rules()
