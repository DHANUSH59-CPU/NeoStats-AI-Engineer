"""Talk-to-data endpoint: natural-language question -> SQL -> readable answer."""
from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import ChatRequest
from src.talk_to_data import prompt_templates as pt

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    """Answer a data question. Returns the SQL, rows, and prose answer (auditable)."""
    from src.talk_to_data.nl_to_sql import ask
    return ask(req.question)


@router.get("/chat/examples")
def examples() -> dict:
    """Suggested questions for the UI."""
    return {"examples": pt.example_questions()}
