"""Explainability endpoint: SHAP risk drivers for one applicant."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import Applicant

router = APIRouter(prefix="/api", tags=["explain"])


@router.post("/explain")
def explain(applicant: Applicant) -> dict:
    """Return top features pushing this applicant's risk up and down (SHAP)."""
    try:
        from src.explain.shap_explainer import explain_one
        return explain_one(applicant.to_raw())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")
