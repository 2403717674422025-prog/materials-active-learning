"""Pydantic schemas for experiment endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Shared config: silence "model_" namespace warning and allow ORM mode
_orm_cfg = ConfigDict(from_attributes=True, protected_namespaces=())
_base_cfg = ConfigDict(protected_namespaces=())


class ExperimentCreate(BaseModel):
    model_config = _base_cfg

    dataset_id: str
    name: Optional[str] = None
    strategy: str = Field(
        default="puda",
        pattern="^(random|uncertainty|ei|ucb|puda)$",
        description="Acquisition strategy",
    )
    model_type: str = Field(
        default="gp",
        pattern="^(gp|rf|xgb|ensemble)$",
        description="Surrogate model type",
    )
    initial_samples: int = Field(default=100, ge=10, le=1000)
    batch_size: int = Field(default=20, ge=1, le=200)
    total_budget: int = Field(default=300, ge=20, le=5000)
    random_seed: int = Field(default=42, ge=0)
    target_column: str = Field(default="gap expt")
    xi: float = Field(default=0.01, ge=0.0, le=1.0)
    kappa: float = Field(default=2.0, ge=0.0, le=10.0)
    alpha: float = Field(default=0.40, ge=0.0, le=1.0)
    beta: float = Field(default=0.35, ge=0.0, le=1.0)
    gamma: float = Field(default=0.25, ge=0.0, le=1.0)
    use_pca: bool = False
    pca_n_components: int = Field(default=50, ge=5, le=200)

    @field_validator("total_budget")
    @classmethod
    def budget_gt_initial(cls, v, values):
        initial = values.data.get("initial_samples", 100)
        if v <= initial:
            raise ValueError("total_budget must be greater than initial_samples")
        return v


class ExperimentResponse(BaseModel):
    model_config = _orm_cfg

    id: str
    name: Optional[str] = None
    dataset_id: str
    strategy: str
    model_type: str
    initial_samples: int
    batch_size: int
    total_budget: int
    current_budget_used: int
    current_iteration: int
    best_found_value: Optional[float] = None
    best_found_id: Optional[str] = None
    global_maximum: Optional[float] = None
    status: str
    created_at: datetime
    updated_at: datetime


class ExperimentStatus(BaseModel):
    experiment_id: str
    status: str
    current_iteration: int
    budget_used: int
    budget_remaining: int
    budget_total: int
    budget_pct: float
    best_found_value: Optional[float] = None
    best_found_id: Optional[str] = None
    global_maximum: Optional[float] = None
    latest_rmse: Optional[float] = None
    latest_r2: Optional[float] = None
    latest_regret: Optional[float] = None


class IterationRequest(BaseModel):
    pass


class IterationResponse(BaseModel):
    model_config = _base_cfg

    iteration: int
    experiment_id: str
    strategy: str
    model_type: str
    budget_used: int
    budget_remaining: int
    selected_candidate_ids: List[str]
    selected_true_values: List[float]
    test_mae: float
    test_rmse: float
    test_r2: float
    best_found_value: float
    best_found_id: str
    regret: float
    global_maximum: float
    training_pool_maximum: float
    mean_uncertainty_selected: float
    diversity_score: float
    iteration_time_seconds: float


class IterationHistoryItem(BaseModel):
    model_config = _orm_cfg

    iteration: int
    budget_used: int
    budget_remaining: int
    test_mae: Optional[float] = None
    test_rmse: Optional[float] = None
    test_r2: Optional[float] = None
    best_found_value: Optional[float] = None
    best_found_id: Optional[str] = None
    regret: Optional[float] = None
    diversity_score: Optional[float] = None
    n_candidates_selected: int = 0
    iteration_time_seconds: Optional[float] = None
    timestamp: datetime
