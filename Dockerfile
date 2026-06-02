# ---- Credit Risk Intelligence Platform image -----------------------------
# Slim Python base; LightGBM needs libgomp at runtime.
FROM python:3.10-slim

# Avoid interactive prompts; make Python logs flush immediately.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Runtime system deps: libgomp1 (LightGBM), curl (healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code (dataset & models are mounted at runtime, not baked in).
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY frontend/ ./frontend/
COPY sql/ ./sql/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fs http://localhost:8000/api/health || exit 1

# On start: build any missing artifacts (DB/model/EDA/rules) from the mounted
# dataset, then launch the API which also serves the frontend.
CMD ["sh", "-c", "python -m src.bootstrap && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"]
