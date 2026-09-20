"""Pydantic schemas for the Loan Approval Prediction API (plan section 20).

The field names intentionally mirror the dataset columns (``No_of_dependents``,
``Income_annum``, ...) so the JSON contract documented in the plan is honoured.
"""

from __future__ import annotations

from datetime import datetime, timezone

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoanApplication(BaseModel):
    """Applicant information sent by the frontend to ``POST /predict``."""

    applicant_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the person the application is for",
    )

    No_of_dependents: int = Field(
        ..., ge=0, le=10, description="Number of dependents"
    )
    Education: Literal["Graduate", "Not Graduate"] = Field(
        ..., description="Graduate / Not Graduate"
    )
    Self_employed: Literal["Yes", "No"] = Field(
        ..., description="Self employed (Yes / No)"
    )
    Income_annum: int = Field(..., ge=0, description="Annual income")
    Loan_amount: int = Field(..., ge=0, description="Requested loan amount")
    Loan_term: int = Field(..., gt=0, description="Loan duration in years")
    Cibil_score: int = Field(..., ge=300, le=900, description="CIBIL credit score")
    # The published dataset contains a handful of negative residential asset
    # values (-100000), so this field is left unconstrained on purpose.
    Residential_assets_value: int = Field(..., description="Residential assets")
    Commercial_assets_value: int = Field(..., ge=0, description="Commercial assets")
    Luxury_assets_value: int = Field(..., ge=0, description="Luxury assets")
    Bank_asset_value: int = Field(..., ge=0, description="Bank assets")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

    def to_features(self) -> dict:
        """Map the API field names onto the pipeline's training feature names.

        Lower-casing ``No_of_dependents`` yields ``no_of_dependents`` which is
        exactly the column name the model was trained on.
        """
        features = self.model_dump(exclude={"applicant_name"})
        return {name.lower(): value for name, value in features.items()}


class PredictionResponse(BaseModel):
    """Response returned by ``POST /predict``."""

    result: Literal["Approved", "Rejected"] = Field(
        ..., description="Predicted loan status"
    )
    probability: float = Field(
        ..., ge=0, le=1, description="Probability of the predicted class"
    )


class HealthResponse(BaseModel):
    """Response returned by ``GET /health``."""

    status: str
    model_name: str
    model_loaded: bool


class PredictionLogResponse(BaseModel):
    """A single logged prediction returned by ``GET /history``."""

    # ``from_attributes`` lets Pydantic build this straight from an ORM object.
    model_config = ConfigDict(from_attributes=True)

    id: int
    applicant_name: str
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int
    prediction: Literal["Approved", "Rejected"]
    probability: float
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _attach_utc(cls, value: datetime) -> datetime:
        """SQLite drops ``tzinfo``, so re-attach UTC before serialising.

        Without this the JSON has no offset and browsers would interpret the
        UTC timestamp as local time.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class HistoryResponse(BaseModel):
    """Response returned by ``GET /history``."""

    count: int = Field(..., description="Number of submissions returned")
    submissions: list[PredictionLogResponse]

