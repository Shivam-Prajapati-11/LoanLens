"""Database setup for the Loan Approval Prediction app (Phase B).

Uses SQLite through SQLAlchemy 2.0 - zero setup, no separate database server.
The database file lives at the project root as ``predictions.db`` whenever that
directory is writable.

Serverless hosts (Vercel, AWS Lambda, ...) mount the deployed bundle read-only,
so the file automatically falls back to the system temp directory. Without that
fallback ``POST /predict`` dies on ``sqlite3.OperationalError: attempt to write
a readonly database`` and the browser only sees an opaque HTTP 500.

Set ``DATABASE_URL`` to a real database (for example Render PostgreSQL) when the
prediction history has to survive restarts.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# --------------------------------------------------------------------------- #
# Engine / session factory
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "predictions.db"


def _directory_is_writable(directory: Path) -> bool:
    """Return ``True`` only when a brand-new file can really be created there.

    A permission flag is not enough (serverless bundles are mounted read-only
    *after* the permissions check passes), so this writes and deletes a probe.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".loanlens_probe"):
            return True
    except OSError:
        return False


def _default_sqlite_path() -> Path:
    """Pick the project root when it is writable, else a temp-directory file.

    ``DATABASE_URL`` always wins over this helper - use it to point the API at a
    durable database instead of an ephemeral SQLite file.
    """
    if _directory_is_writable(BASE_DIR):
        if not DATABASE_PATH.exists() or os.access(DATABASE_PATH, os.W_OK):
            return DATABASE_PATH
    return Path(tempfile.gettempdir()) / "predictions.db"


def _normalise_url(url: str) -> str:
    """SQLAlchemy 2.x rejects the legacy ``postgres://`` scheme."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url


# ``DATABASE_URL`` can be overridden (the tests point it at a throwaway file).
DATABASE_URL = _normalise_url(
    os.environ.get("DATABASE_URL") or f"sqlite:///{_default_sqlite_path()}"
)

# ``check_same_thread`` is a SQLite-only argument - passing it to PostgreSQL
# (or any other dialect) raises, so it is applied conditionally.
CONNECT_ARGS: dict = (
    {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(DATABASE_URL, connect_args=CONNECT_ARGS, future=True)

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
