"""ORM models for experiments and their iterations."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(256), nullable=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), nullable=False)
    # Strategy + model
    strategy: Mapped[str] = mapped_column(String(32), default="puda")
    model_type: Mapped[str] = mapped_column(String(32), default="gp")
    # Budget configuration
    initial_samples: Mapped[int] = mapped_column(Integer, default=100)
    batch_size: Mapped[int] = mapped_column(Integer, default=20)
    total_budget: Mapped[int] = mapped_column(Integer, default=300)
    current_budget_used: Mapped[int] = mapped_column(Integer, default=0)
    current_iteration: Mapped[int] = mapped_column(Integer, default=0)
    # Acquisition hyperparameters
    xi: Mapped[float] = mapped_column(Float, default=0.01)
    kappa: Mapped[float] = mapped_column(Float, default=2.0)
    alpha: Mapped[float] = mapped_column(Float, default=0.40)
    beta: Mapped[float] = mapped_column(Float, default=0.35)
    gamma: Mapped[float] = mapped_column(Float, default=0.25)
    # Preprocessing
    use_pca: Mapped[bool] = mapped_column(default=False)
    pca_n_components: Mapped[int] = mapped_column(Integer, default=50)
    # Reproducibility
    random_seed: Mapped[int] = mapped_column(Integer, default=42)
    target_column: Mapped[str] = mapped_column(String(128), default="gap expt")
    # Current best result
    best_found_value: Mapped[float] = mapped_column(Float, nullable=True)
    best_found_id: Mapped[str] = mapped_column(String(256), nullable=True)
    global_maximum: Mapped[float] = mapped_column(Float, nullable=True)
    # Status
    status: Mapped[str] = mapped_column(String(32), default="created")
    # Serialized engine state (pickle path)
    state_path: Mapped[str] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    iterations: Mapped[list["ExperimentIteration"]] = relationship(
        "ExperimentIteration", back_populates="experiment", cascade="all, delete-orphan"
    )


class ExperimentIteration(Base):
    __tablename__ = "experiment_iterations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id"), nullable=False)
    iteration_num: Mapped[int] = mapped_column(Integer, nullable=False)
    # Metrics
    test_mae: Mapped[float] = mapped_column(Float, nullable=True)
    test_rmse: Mapped[float] = mapped_column(Float, nullable=True)
    test_r2: Mapped[float] = mapped_column(Float, nullable=True)
    best_found_value: Mapped[float] = mapped_column(Float, nullable=True)
    best_found_id: Mapped[str] = mapped_column(String(256), nullable=True)
    regret: Mapped[float] = mapped_column(Float, nullable=True)
    diversity_score: Mapped[float] = mapped_column(Float, nullable=True)
    # Budget
    budget_used: Mapped[int] = mapped_column(Integer, nullable=True)
    budget_remaining: Mapped[int] = mapped_column(Integer, nullable=True)
    # Selected candidates (JSON array of IDs)
    selected_candidate_ids: Mapped[list] = mapped_column(JSON, default=list)
    selected_true_values: Mapped[list] = mapped_column(JSON, default=list)
    # Timing
    iteration_time_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="iterations")


class TuningRun(Base):
    """Persisted metadata and compact trial history for one Optuna run."""

    __tablename__ = "tuning_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id"), nullable=False, index=True
    )
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    method: Mapped[str] = mapped_column(String(32), default="optuna_tpe")
    objective: Mapped[str] = mapped_column(String(32), default="validation_rmse")
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    best_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    best_validation_rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_validation_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_validation_r2: Mapped[float | None] = mapped_column(Float, nullable=True)
    trial_count: Mapped[int] = mapped_column(Integer, default=0)
    completed_trials: Mapped[int] = mapped_column(Integer, default=0)
    failed_trials: Mapped[int] = mapped_column(Integer, default=0)
    pruned_trials: Mapped[int] = mapped_column(Integer, default=0)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    timeout_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    training_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    validation_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    split_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    search_space_version: Mapped[str] = mapped_column(String(32), default="rf-gp-v1")
    trial_history: Mapped[list] = mapped_column(JSON, default=list)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    applied_to_experiment: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
