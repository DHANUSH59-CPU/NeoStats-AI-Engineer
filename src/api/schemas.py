"""Pydantic request/response models for the API.

Applicant uses a small set of the most decision-relevant fields. Every field is
optional: the preprocessor imputes anything missing, so the UI form can stay
short while the model still scores a full feature vector.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Applicant(BaseModel):
    """Raw applicant input from the UI (a subset of application_train columns)."""

    AMT_INCOME_TOTAL: Optional[float] = Field(None, description="Annual income")
    AMT_CREDIT: Optional[float] = Field(None, description="Loan amount")
    AMT_ANNUITY: Optional[float] = Field(None, description="Loan annuity")
    AMT_GOODS_PRICE: Optional[float] = None
    DAYS_BIRTH: Optional[int] = Field(None, description="Age in days (negative)")
    DAYS_EMPLOYED: Optional[int] = Field(None, description="Days employed (negative)")
    CNT_CHILDREN: Optional[int] = 0
    CNT_FAM_MEMBERS: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)
    CODE_GENDER: Optional[str] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None

    def to_raw(self) -> dict[str, Any]:
        """Drop None values so the preprocessor imputes them."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural-language question")
