"""Immutable persisted records for reproducible strategy benchmark runs."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    configuration: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    results: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
