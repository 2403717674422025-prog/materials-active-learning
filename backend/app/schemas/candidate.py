"""Pydantic schemas for candidate predictions and selections."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CandidatePrediction(BaseModel):
    candidate_id: str
    pool_index: int
    predicted_mean: float
    predicted_std: float
    lower_cb: Optional[float] = None
    upper_cb: Optional[float] = None
    acquisition_score: float
    is_selected: bool
    selection_rank: Optional[int] = None
    uncertainty_label: str = "Unknown"  # Low | Medium | High
    breakdown: Dict[str, Any] = {}

    model_config = {"from_attributes": True}


class CandidateListResponse(BaseModel):
    experiment_id: str
    iteration: int
    total_candidates: int
    selected_count: int
    candidates: List[CandidatePrediction]
    # Pagination
    page: int = 1
    page_size: int = 50
    total_pages: int = 1


class CandidateDetail(BaseModel):
    """Detailed view for the Selection Explainer page (Page 6)."""
    candidate_id: str
    iteration: int
    # Core prediction
    predicted_mean: float
    predicted_std: float
    lower_cb: Optional[float] = None
    upper_cb: Optional[float] = None
    # Acquisition score breakdown
    strategy: str
    acquisition_score: float
    predicted_performance: Optional[float] = None
    uncertainty: Optional[float] = None
    exploration_contribution: Optional[float] = None
    exploitation_contribution: Optional[float] = None
    diversity_contribution: Optional[float] = None
    formula: Optional[str] = None
    selection_reason: str = ""
    # Context
    is_selected: bool
    selection_rank: Optional[int] = None
    # Full breakdown dict
    full_breakdown: Dict[str, Any] = {}


class DiscoveredMaterial(BaseModel):
    """Row in the Material Discovery page (Page 9)."""
    rank: int
    candidate_id: str
    true_value: float
    iteration_discovered: int
    strategy: str
    budget_at_discovery: int
    predicted_mean: float
    predicted_std: float
    acquisition_score: float
    feature_preview: Dict[str, float] = {}
