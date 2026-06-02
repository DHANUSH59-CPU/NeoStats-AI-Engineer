"""Prediction endpoint: score an applicant -> default probability + risk band."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import Applicant

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict")
def predict(applicant: Applicant) -> dict:
    """Return default probability, risk band, and the model's headline metrics."""
    try:
        from src.ml.predict import predict_one
        return predict_one(applicant.to_raw())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
