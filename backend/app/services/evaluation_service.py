"""Evaluation service: build comparison charts from multiple experiments."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

from backend.app.models.experiment import Experiment, ExperimentIteration
from backend.app.models.result import ExperimentalResult
from backend.app.services.dataset_service import get_dataset_by_id, load_dataset_arrays
from backend.ml.data.preprocessing import initial_split
from backend.ml.evaluation.comparison import StrategyComparison, build_comparison_from_db_records
from backend.ml.evaluation.metrics import (
    cumulative_regret,
    discovery_curve_with_initial,
    normalized_area_under_discovery_curve,
)

logger = logging.getLogger(__name__)


def build_experiment_comparison(
    db: Session,
    experiment_ids: List[str],
) -> dict:
    """
    Given a list of experiment IDs (one per strategy), build full comparison charts.
    """
    records = []

    for eid in experiment_ids:
        exp = db.query(Experiment).filter(Experiment.id == eid).first()
        if not exp:
            logger.warning("Experiment %s not found, skipping", eid)
            continue

        iters = (
            db.query(ExperimentIteration)
            .filter(ExperimentIteration.experiment_id == eid)
            .order_by(ExperimentIteration.iteration_num)
            .all()
        )

        if not iters:
            continue

        # Build curves from iteration records
        budget_curve = [it.budget_used for it in iters]
        rmse_curve = [it.test_rmse or 0.0 for it in iters]
        r2_curve = [it.test_r2 or 0.0 for it in iters]
        best_found_vals = [it.best_found_value or 0.0 for it in iters]
        diversity_curve = [it.diversity_score or 0.0 for it in iters]

        # Acquired values in order (for top-k discovery rate)
        all_results = (
            db.query(ExperimentalResult)
            .filter(ExperimentalResult.experiment_id == eid)
            .order_by(
                ExperimentalResult.iteration_num,
                ExperimentalResult.acquired_at,
            )
            .all()
        )
        acquired_in_order = [r.true_value for r in all_results]

        # Get all pool values for top-k reference
        dataset = get_dataset_by_id(db, exp.dataset_id)
        try:
            X, y, candidate_ids = load_dataset_arrays(dataset)
            all_pool_values = y.values
            global_maximum = float(y.max())
            split = initial_split(X, y, candidate_ids, test_size=0.20, random_state=exp.random_seed)
            initial_n = min(exp.initial_samples, len(split["pool_y"]) - 1)
            seed_rng = np.random.RandomState(exp.random_seed)
            seed_indices = seed_rng.choice(len(split["pool_y"]), size=initial_n, replace=False)
            initial_best = float(split["pool_y"].iloc[seed_indices].max())
        except Exception:
            all_pool_values = np.array([])
            global_maximum = float(exp.global_maximum or 0.0)
            initial_n = int(exp.initial_samples)
            initial_best = float(best_found_vals[0])

        discovery_budgets, discovery_values = discovery_curve_with_initial(
            initial_best, acquired_in_order, initial_n,
        )
        naudc = normalized_area_under_discovery_curve(
            discovery_values, global_maximum, budgets=discovery_budgets,
        )

        records.append({
            "strategy": exp.strategy,
            "budget_curve": budget_curve,
            "rmse_curve": rmse_curve,
            "r2_curve": r2_curve,
            "best_found_curve": discovery_values,
            "discovery_budget_curve": discovery_budgets,
            "diversity_curve": diversity_curve,
            "acquired_values_in_order": acquired_in_order,
            "global_maximum": global_maximum,
            "all_pool_values": all_pool_values,
        })

    if not records:
        return {}

    charts = build_comparison_from_db_records(records)
    # Discovery and regret are candidate-level; model-quality series remain at
    # their recorded post-batch checkpoints. Keep each x-axis paired to its data.
    discovery_by_strategy = {r["strategy"]: r for r in records}
    for series in charts.get("discovery_curve", []):
        record = discovery_by_strategy.get(series["strategy"])
        if record:
            series["budget"] = record["discovery_budget_curve"]
            series["budget_unit"] = "cumulative_candidates"
            series["best_found"] = record["best_found_curve"]
    for series in charts.get("regret_curve", []):
        record = discovery_by_strategy.get(series["strategy"])
        if record:
            series["budget"] = record["discovery_budget_curve"]
            series["budget_unit"] = "cumulative_candidates"
            series["regret"] = cumulative_regret(
                record["best_found_curve"], record["global_maximum"],
            )
    for series_key in ("rmse_curve", "diversity_curve"):
        for series in charts.get(series_key, []):
            series["budget_unit"] = "cumulative_candidates"
    return charts


def get_experiment_evaluation_summary(db: Session, experiment_id: str) -> dict:
    """Return the evaluation summary for a single experiment."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")

    iters = (
        db.query(ExperimentIteration)
        .filter(ExperimentIteration.experiment_id == experiment_id)
        .order_by(ExperimentIteration.iteration_num)
        .all()
    )

    if not iters:
        return {"experiment_id": experiment_id, "status": "no_iterations"}

    dataset = get_dataset_by_id(db, exp.dataset_id)
    try:
        X, y, candidate_ids = load_dataset_arrays(dataset)
        global_max = float(y.max())
        split = initial_split(X, y, candidate_ids, test_size=0.20, random_state=exp.random_seed)
        initial_n = min(exp.initial_samples, len(split["pool_y"]) - 1)
        seed_rng = np.random.RandomState(exp.random_seed)
        seed_indices = seed_rng.choice(len(split["pool_y"]), size=initial_n, replace=False)
        initial_best = float(split["pool_y"].iloc[seed_indices].max())
    except Exception:
        global_max = float(exp.global_maximum or 0.0)
        initial_n = int(exp.initial_samples)
        initial_best = float(iters[0].best_found_value or 0.0)

    acquired = (
        db.query(ExperimentalResult)
        .filter(ExperimentalResult.experiment_id == experiment_id)
        .order_by(ExperimentalResult.iteration_num, ExperimentalResult.acquired_at, ExperimentalResult.id)
        .all()
    )
    discovery_budgets, discovery_values = discovery_curve_with_initial(
        initial_best, [float(row.true_value) for row in acquired], initial_n,
    )
    naudc = normalized_area_under_discovery_curve(
        discovery_values, global_max, budgets=discovery_budgets,
    )

    last = iters[-1]
    return {
        "experiment_id": experiment_id,
        "strategy": exp.strategy,
        "model_type": exp.model_type,
        "final_mae": last.test_mae,
        "final_rmse": last.test_rmse,
        "final_r2": last.test_r2,
        "final_regret": global_max - last.best_found_value
        if last.best_found_value is not None else last.regret,
        "naudc": naudc,
        "best_found_value": last.best_found_value,
        "global_maximum": global_max,
        "budget_used": last.budget_used,
        "discovery_budget": discovery_budgets,
        "discovery_curve": discovery_values,
        "budget_unit": "cumulative_candidates",
        "n_iterations": len(iters),
    }
