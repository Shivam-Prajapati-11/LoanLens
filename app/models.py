"""SQLAlchemy ORM models for the Loan Approval Prediction app (Phase B).

``PredictionLog`` stores one row for every call to ``POST /predict`` so the
history dashboard can show past submissions.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class PredictionLog(Base):
    """A single logged loan-approval prediction."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    applicant_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # --- model inputs -----------------------------------------------------
    no_of_dependents: Mapped[int] = mapped_column(Integer, nullable=False)
    education: Mapped[str] = mapped_column(String(20), nullable=False)
    self_employed: Mapped[str] = mapped_column(String(5), nullable=False)
    income_annum: Mapped[int] = mapped_column(BigInteger, nullable=False)
    loan_amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    loan_term: Mapped[int] = mapped_column(Integer, nullable=False)
    cibil_score: Mapped[int] = mapped_column(Integer, nullable=False)
    residential_assets_value: Mapped[int] = mapped_column(BigInteger, nullable=False)
    commercial_assets_value: Mapped[int] = mapped_column(BigInteger, nullable=False)
    luxury_assets_value: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bank_asset_value: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # --- model output -----------------------------------------------------
    prediction: Mapped[str] = mapped_column(String(10), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"PredictionLog(id={self.id!r}, applicant_name={self.applicant_name!r}, "
            f"prediction={self.prediction!r})"
        )
