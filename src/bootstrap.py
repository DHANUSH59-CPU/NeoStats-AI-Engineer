"""Startup bootstrap: make sure all runtime artifacts exist.

Called once when the container starts (before uvicorn). It regenerates anything
missing from the mounted dataset, so a fresh `docker-compose up` becomes fully
self-contained:
  * models/credit.db        (SQLite for the chatbot) — large, not committed
  * models/model.pkl        (trained model) — committed, but retrained if absent
  * models/eda_summary.json (EDA for the UI)
  * models/rules.json       (derived rules)

Each step is skipped if its artifact already exists, so restarts are fast.
"""
from __future__ import annotations

from src.utils import config
from src.utils.logger import get_logger

log = get_logger("bootstrap")


def main() -> None:
    config.ensure_dirs()

    if not config.APPLICATION_TRAIN.exists():
        log.error("Dataset not found at %s. Mount the Home Credit CSVs there "
                  "(see README).", config.DATA_DIR)
        # Don't crash: the API can still serve whatever artifacts exist.
        return

    if not config.DB_PATH.exists():
        log.info("Building SQLite DB (chatbot)...")
        from src.data.loader import build_sqlite_db
        build_sqlite_db()
    else:
        log.info("SQLite DB present — skipping.")

    if not config.MODEL_PATH.exists():
        log.info("Training model (no artifact found)...")
        from src.ml.train import train
        train()
    else:
        log.info("Model artifact present — skipping training.")

    if not config.EDA_SUMMARY_PATH.exists():
        log.info("Generating EDA summary...")
        from notebooks.eda import run_eda
        run_eda()
    else:
        log.info("EDA summary present — skipping.")

    if not config.RULES_PATH.exists():
        log.info("Deriving rules...")
        from src.rules.derive_rules import derive_rules
        derive_rules()
    else:
        log.info("Rules present — skipping.")

    log.info("Bootstrap complete.")


if __name__ == "__main__":
    main()
