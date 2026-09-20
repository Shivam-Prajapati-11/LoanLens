"""Database setup for the Loan Approval Prediction app (Phase B).

Uses SQLite through SQLAlchemy 2.0 - zero setup, no separate database server.
The database file lives at the project root as ``predictions.db``.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# --------------------------------------------------------------------------- #
# Engine / session factory
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "predictions.db"
# ``DATABASE_URL`` can be overridden (the tests point it at a throwaway file).
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")

# ``check_same_thread=False`` is required because FastAPI serves requests from a
# thread pool while the connection is created on the main thread.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base every ORM model inherits from."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create the tables if they do not exist yet.

    Importing ``app.models`` registers the ORM classes on ``Base.metadata``
    before ``create_all`` runs.
    """
    from app import models  # noqa: F401  (import registers the models)

    Base.metadata.create_all(bind=engine)
