"""Pydantic schemas for evaluation and comparison endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StrategySeriesData(BaseModel):
    strategy: str
    label: str
    color: str
    budget: List[int]
    values: List[float]


class ComparisonCharts(BaseModel):
    discovery_curve: List[Dict[str, Any]]
    rmse_curve: List[Dict[str, Any]]
    regret_curve: List[Dict[str, Any]]
    diversity_curve: List[Dict[str, Any]]
    summary_table: List[Dict[str, Any]]
    ranking: Dict[str, Any]
    global_optimum: float
    strategies: List[str]


class AblationRow(BaseModel):
    variant: str
    label: str
    naudc: float
    final_rmse: float
    final_r2: float
    final_regret: float


class AblationSummary(BaseModel):
    ablation_table: List[AblationRow]
    best_variant: Optional[str] = None


class CalibrationData(BaseModel):
    ece: float
    coverage_90pct: float
    coverage_95pct: float
    expected_coverage: List[float]
    observed_coverage: List[float]


class EvaluationSummary(BaseModel):
    model_config = {"protected_namespaces": ()}
    experiment_id: str
    strategy: str
    model_type: str
    final_mae: Optional[float] = None
    final_rmse: Optional[float] = None
    final_r2: Optional[float] = None
    final_regret: Optional[float] = None
    naudc: Optional[float] = None
    best_found_value: Optional[float] = None
    global_maximum: Optional[float] = None
    top10_discovery_rate: Optional[float] = None
    calibration: Optional[CalibrationData] = None
    budget_used: Optional[int] = None
    n_iterations: Optional[int] = None
