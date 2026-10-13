"""ORM models for candidate predictions and experimental results."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class CandidatePredictionRecord(Base):
    __tablename__ = "candidate_predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id"), nullable=False)
    iteration_num: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_id: Mapped[str] = mapped_column(String(256), nullable=False)
    pool_index: Mapped[int] = mapped_column(Integer, nullable=True)
    # Predictions
    predicted_mean: Mapped[float] = mapped_column(Float, nullable=True)
    predicted_std: Mapped[float] = mapped_column(Float, nullable=True)
    lower_cb: Mapped[float] = mapped_column(Float, nullable=True)
    upper_cb: Mapped[float] = mapped_column(Float, nullable=True)
    # Acquisition
    acquisition_score: Mapped[float] = mapped_column(Float, nullable=True)
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False)
    selection_rank: Mapped[int] = mapped_column(Integer, nullable=True)
    # PUDA breakdown (stored as JSON)
    breakdown: Mapped[dict] = mapped_column(JSON, default=dict)


class ExperimentalResult(Base):
    __tablename__ = "experimental_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id"), nullable=False)
    iteration_num: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_id: Mapped[str] = mapped_column(String(256), nullable=False)
    pool_index: Mapped[int] = mapped_column(Integer, nullable=True)
    true_value: Mapped[float] = mapped_column(Float, nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
