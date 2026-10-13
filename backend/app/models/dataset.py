"""ORM model for uploaded/loaded datasets."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=True)
    source: Mapped[str] = mapped_column(String(256), nullable=True)   # e.g. "matbench_expt_gap"
    num_samples: Mapped[int] = mapped_column(Integer, nullable=True)
    num_features: Mapped[int] = mapped_column(Integer, nullable=True)
    target_column: Mapped[str] = mapped_column(String(128), nullable=True)
    composition_column: Mapped[str] = mapped_column(String(128), default="composition")
    feature_columns: Mapped[list] = mapped_column(JSON, default=list)
    global_maximum: Mapped[float] = mapped_column(nullable=True)
    global_minimum: Mapped[float] = mapped_column(nullable=True)
    preprocessing_config: Mapped[dict] = mapped_column(JSON, default=dict)
    # Path to cached featurized data
    cache_path: Mapped[str] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|ready|error
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    upload_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ready_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
