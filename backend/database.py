"""
SQLAlchemy database setup.

Supports:
  - SQLite (default for local dev, no extra services needed)
  - PostgreSQL (production, set DATABASE_URL env var)

Uses synchronous SQLAlchemy for simplicity (no async complexity needed
given that ML training jobs are CPU-bound and run synchronously anyway).
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import get_settings

settings = get_settings()

# For SQLite: add check_same_thread=False to allow multi-threaded access
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db():
    """FastAPI dependency: yields a DB session and ensures it is closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables():
    """Create all tables. Called on app startup."""
    # Import all models so SQLAlchemy registers them before create_all
    from backend.app.models import benchmark, dataset, experiment, result  # noqa: F401
    from backend.app.models.experiment import TuningRun  # noqa: F401
    Base.metadata.create_all(bind=engine)
