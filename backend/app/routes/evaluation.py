"""Evaluation and comparison routes."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.schemas.evaluation import ComparisonCharts, EvaluationSummary
from backend.app.services.evaluation_service import (
    build_experiment_comparison,
    get_experiment_evaluation_summary,
)
from backend.database import get_db

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


# ── CRITICAL: static routes must come BEFORE parameterized routes ─────────────
# Bug fix: previously /compare/strategies was declared after /{experiment_id},
# causing FastAPI to match "compare" as an experiment_id.

@router.get("/compare/strategies")
def compare_strategies(
    experiment_ids: List[str] = Query(..., description="Experiment IDs to compare"),
    db: Session = Depends(get_db),
):
    """
    Compare multiple experiments (one per strategy) and return chart data.
    Pass multiple experiment_ids as repeated query params:
      ?experiment_ids=abc&experiment_ids=def
    """
    if len(experiment_ids) < 1:
        raise HTTPException(
            status_code=422,
            detail="Provide at least 1 experiment ID.",
        )
    try:
        charts = build_experiment_comparison(db, experiment_ids)
        return charts
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{experiment_id}", response_model=EvaluationSummary)
def get_evaluation(experiment_id: str, db: Session = Depends(get_db)):
    """Get evaluation summary for a single experiment."""
    try:
        data = get_experiment_evaluation_summary(db, experiment_id)
        return EvaluationSummary(**{
            k: data.get(k) for k in EvaluationSummary.model_fields
        })
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
