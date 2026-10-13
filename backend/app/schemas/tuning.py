"""Request and response schemas for hyperparameter tuning."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TuningStartRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str = Field(pattern="^(rf|gp)$")
    n_trials: int = Field(default=20, ge=1, le=100)
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)
    timeout_seconds: int | None = Field(default=900, ge=1, le=3600)


class TuningRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: str
    experiment_id: str
    model_type: str
    method: str
    objective: str
    status: str
    best_params: dict[str, Any] | None
    best_validation_rmse: float | None
    best_validation_mae: float | None
    best_validation_r2: float | None
    trial_count: int
    completed_trials: int
    failed_trials: int
    pruned_trials: int
    random_seed: int
    timeout_seconds: int | None
    training_size: int | None
    validation_size: int | None
    dataset_fingerprint: str | None
    split_metadata: dict[str, Any] | None
    search_space_version: str
    trial_history: list[dict[str, Any]]
    error_summary: str | None
    applied_to_experiment: bool
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
