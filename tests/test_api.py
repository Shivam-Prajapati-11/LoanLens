"""Smoke tests for the FastAPI backend (Phases 8 & 9).

Run them with either::

    python -m tests.test_api

or, if pytest is installed::

    pytest tests/test_api.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Make the project root importable when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Isolate the tests from the real predictions.db: point the app at a throwaway
# SQLite file created in a temp directory for this run. Must happen BEFORE
# importing app.main, because the engine and tables are created at import time.
_TEMP_DIR = Path(tempfile.mkdtemp(prefix="loan_prediction_tests_"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TEMP_DIR / 'predictions_test.db'}"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

# A strong applicant: high income, good CIBIL score -> should be Approved.
APPROVED_APPLICATION = {
    "applicant_name": "Ravi Kumar",
    "No_of_dependents": 2,
    "Education": "Graduate",
    "Self_employed": "No",
    "Income_annum": 5000000,
    "Loan_amount": 20000000,
    "Loan_term": 10,
    "Cibil_score": 750,
    "Residential_assets_value": 10000000,
    "Commercial_assets_value": 5000000,
    "Luxury_assets_value": 3000000,
    "Bank_asset_value": 5000000,
}

# A weak applicant: low income, poor CIBIL score -> should be Rejected.
REJECTED_APPLICATION = {
    "applicant_name": "Meera Sharma",
    "No_of_dependents": 1,
    "Education": "Not Graduate",
    "Self_employed": "Yes",
    "Income_annum": 1000000,
    "Loan_amount": 30000000,
    "Loan_term": 20,
    "Cibil_score": 350,
    "Residential_assets_value": 0,
    "Commercial_assets_value": 0,
    "Luxury_assets_value": 300000,
    "Bank_asset_value": 0,
}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_index_page_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Loan Approval Prediction" in response.text


def test_predict_approves_strong_application() -> None:
    response = client.post("/predict", json=APPROVED_APPLICATION)
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == "Approved"
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_weak_application() -> None:
    response = client.post("/predict", json=REJECTED_APPLICATION)
    assert response.status_code == 200
    assert response.json()["result"] == "Rejected"


def test_predict_validates_out_of_range_payload() -> None:
    invalid = dict(APPROVED_APPLICATION, Cibil_score=50)
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_requires_applicant_name() -> None:
    payload = {k: v for k, v in APPROVED_APPLICATION.items() if k != "applicant_name"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_is_logged_in_history() -> None:
    before = client.get("/api/history").json()["count"]

    payload = dict(APPROVED_APPLICATION, applicant_name="History Tester")
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = client.get("/api/history").json()
    assert body["count"] == before + 1

    newest = body["submissions"][0]
    assert newest["applicant_name"] == "History Tester"
    assert newest["prediction"] == response.json()["result"]
    assert newest["cibil_score"] == payload["Cibil_score"]
    assert newest["income_annum"] == payload["Income_annum"]
    assert newest["created_at"]


def test_history_page_is_served() -> None:
    response = client.get("/history")
    assert response.status_code == 200
    assert "Prediction History" in response.text


def _run() -> None:
    tests = [
        test_health,
        test_index_page_is_served,
        test_predict_approves_strong_application,
        test_predict_rejects_weak_application,
        test_predict_validates_out_of_range_payload,
        test_predict_requires_applicant_name,
        test_predict_is_logged_in_history,
        test_history_page_is_served,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run()
