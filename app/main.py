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
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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
# CORS
#
# The frontend can be served from this same origin (``/`` -> static/index.html)
# or hosted separately, e.g. a Vercel deployment talking to this API on Render.
# The browser blocks the cross-origin calls without these headers - which looks
# exactly like "Vercel is not connected to Render".
#
# ``ALLOWED_ORIGINS`` (comma separated) overrides the defaults below, so the
# Render dashboard can point at whatever domain the frontend actually uses.
# --------------------------------------------------------------------------- #
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://loan-approval-prediction-self.vercel.app",
]

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Preview deployments get a new ``*.vercel.app`` subdomain on every push.
    allow_origin_regex=r"https://[a-z0-9-]+(\.[a-z0-9-]+)*\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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
    # A storage failure here (read-only filesystem, database outage) must not
    # masquerade as a generic 500: the response names the real cause.
    try:
        db.add(
            PredictionLog(
                applicant_name=application.applicant_name,
                **features,
                prediction=TARGET_LABELS[prediction],
                probability=round(probability, 4),
            )
        )
        db.commit()
    except Exception as error:  # pragma: no cover - infrastructure guard
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction succeeded but could not be stored "
                f"({error}). Check DATABASE_URL / filesystem permissions."
            ),
        )

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
