"""FastAPI application entry point.

Wires together the five platform capabilities behind a small JSON API and serves
the static frontend. Heavy objects (model, SHAP explainer) are loaded once at
startup via the lifespan handler, so requests stay fast.

Run locally:  uvicorn src.api.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers import chat, eda, explain, predict, rules
from src.utils import config
from src.utils.logger import get_logger

log = get_logger("api")

FRONTEND_DIR = config.PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the model + explainer once at startup (best-effort)."""
    try:
        from src.ml.predict import load_artifact
        load_artifact()
        log.info("Model artifact loaded at startup.")
    except Exception as e:  # model may be trained later; don't crash the API
        log.warning("Model not loaded at startup: %s", e)
    yield


app = FastAPI(
    title="Credit Risk Intelligence Platform",
    description="EDA, default-risk scoring, SHAP explainability, rules, and talk-to-data.",
    version="1.0.0",
    lifespan=lifespan,
)

# API routers
app.include_router(eda.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(rules.router)
app.include_router(chat.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_present": config.MODEL_PATH.exists()}


# Serve the static frontend at the root.
if FRONTEND_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=config.APP_HOST, port=config.APP_PORT)
