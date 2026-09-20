"""FastAPI backend for the Loan Approval Prediction system (Phase 8).

The API loads the pipeline saved by ``src/train.py`` and exposes:

* ``POST /predict``  - predict Approved / Rejected for an application
* ``GET  /health``   - liveness probe
* ``GET  /metadata`` - model information and the metrics it was selected on
* ``GET  /api/history`` - every stored prediction (SQLite), newest first
* ``GET  /history``  - the history dashboard page
* ``GET  /``         - the static frontend (Phase 9)
* ``GET  /docs``     - the auto-generated Swagger UI

Run locally with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.models import PredictionLog
from app.schemas import (
    HealthResponse,
    HistoryResponse,
    LoanApplication,
    PredictionLogResponse,
    PredictionResponse,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
STATIC_DIR = BASE_DIR / "static"

# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="LoanLens API",
    description=(
        "**LoanLens** - machine-learning service that predicts whether a loan application "
        "would be **Approved** or **Rejected**, and stores every prediction in SQLite."
    ),
    version="1.0.0",
)

# --------------------------------------------------------------------------- #
# Load the trained pipeline + metadata once, at import time.
# --------------------------------------------------------------------------- #
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run `python -m src.train` before starting the API."
    )

MODEL = joblib.load(MODEL_PATH)
METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

FEATURE_COLUMNS: list[str] = METADATA["feature_columns"]
# {"Rejected": 0, "Approved": 1} -> {0: "Rejected", 1: "Approved"}
TARGET_LABELS: dict[int, str] = {
    int(index): label for label, index in METADATA["target_mapping"].items()
}


# Create the SQLite tables (idempotent). Done at import time instead of in a
# lifespan handler so it also runs under ``fastapi.testclient``, which does not
# fire startup/lifespan events when used directly.
init_db()


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Simple liveness probe used by Docker / Render."""
    return HealthResponse(
        status="ok",
        model_name=METADATA["model_name"],
        model_loaded=True,
    )


@app.get("/metadata", tags=["system"])
def metadata() -> dict:
    """Expose the model metadata (selected model, metrics, feature options)."""
    return METADATA


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    application: LoanApplication,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Run the trained pipeline on a single loan application."""
    features = application.to_features()

    # Build a one-row DataFrame with the same column names/order as training;
    # the pipeline applies the exact preprocessing used during fitting.
    frame = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    try:
        prediction = int(MODEL.predict(frame)[0])
        probability = float(MODEL.predict_proba(frame)[0][prediction])
    except Exception as error:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")

    # Persist the submission so the history dashboard can show it later.
    # ``features`` already uses the same names as the ORM columns.
    db.add(
        PredictionLog(
            applicant_name=application.applicant_name,
            **features,
            prediction=TARGET_LABELS[prediction],
            probability=round(probability, 4),
        )
    )
    db.commit()

    return PredictionResponse(
        result=TARGET_LABELS[prediction],
        probability=round(probability, 4),
    )


@app.get("/api/history", response_model=HistoryResponse, tags=["prediction"])
def history(
    limit: int = Query(200, ge=1, le=1000, description="Maximum rows to return"),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """Return the stored prediction history, newest first.

    NOTE: the plan asked for this JSON endpoint *and* an HTML page at the same
    path (``/history``), which is impossible. The page owns ``/history`` and the
    JSON API lives here at ``/api/history``.
    """
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc(), PredictionLog.id.desc())
        .limit(limit)
        .all()
    )
    return HistoryResponse(
        count=len(rows),
        submissions=[PredictionLogResponse.model_validate(row) for row in rows],
    )


# --------------------------------------------------------------------------- #
# Static frontend (Phase 9). Mounted last so it never shadows the API routes.
# --------------------------------------------------------------------------- #
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/history", include_in_schema=False)
    def history_page() -> FileResponse:
        """Serve the prediction history dashboard."""
        return FileResponse(STATIC_DIR / "history.html")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the single-page frontend."""
        return FileResponse(STATIC_DIR / "index.html")
