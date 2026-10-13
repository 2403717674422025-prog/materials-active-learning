"""
Benchmark service.

Runs controlled multi-strategy experiments under MATCHED conditions:
  - Identical initial labeled set (same random seed + same initial indices)
  - Identical candidate pool
  - Identical total budget and batch size
  - Multiple random seeds (repeats)
  - All four strategies: random, uncertainty, ei, puda

Also runs PUDA ablation study:
  - pred_only, pred_unc, pred_div, full

Results are stored in the DB under a parent Experiment's benchmark record.
"""

from __future__ import annotations

import logging
import hashlib
import json
import platform
import time
import uuid
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn
from sqlalchemy.orm import Session

from backend.app.models.benchmark import BenchmarkRun
from backend.app.models.experiment import Experiment
from backend.app.models.result import CandidatePredictionRecord, ExperimentalResult
from backend.app.services.dataset_service import get_dataset_by_id, load_dataset_arrays
from backend.ml.active_learning.engine import ActiveLearningEngine, ExperimentConfig
from backend.ml.data.preprocessing import initial_split
from backend.ml.evaluation.metrics import (
    cumulative_regret,
    discovery_curve_with_initial,
    expected_calibration_error,
    normalized_area_under_discovery_curve,
    top_k_discovery_rate,
)

logger = logging.getLogger(__name__)

STRATEGIES = ["random", "uncertainty", "ei", "puda"]
DEFAULT_BENCHMARK_SEEDS = [42, 123, 456]
INTERRUPTED_BENCHMARK_TIMEOUT = timedelta(hours=24)
_ACTIVE_BENCHMARK_RUN_IDS: set[str] = set()
_ACTIVE_BENCHMARK_RUN_IDS_LOCK = Lock()
ABLATION_MODES = ["pred_only", "pred_unc", "pred_div", "full"]
ABLATION_CONFIGS = {
    "pred_only": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    "pred_unc":  {"alpha": 0.5, "beta": 0.5, "gamma": 0.0},
    "pred_div":  {"alpha": 0.5, "beta": 0.0, "gamma": 0.5},
    "full":      {"alpha": 0.4, "beta": 0.35,"gamma": 0.25},
}
STRATEGY_COLORS = {
    "random":      "#3A4F64",
    "uncertainty": "#59D5CC",
    "ei":          "#A99BFF",
    "puda":        "#41D98C",
    "pred_only":   "#F06565",
    "pred_unc":    "#EDAB60",
    "pred_div":    "#59D5CC",
    "full":        "#41D98C",
}
STRATEGY_LABELS = {
    "random":      "Random Sampling",
    "uncertainty": "Uncertainty Sampling",
    "ei":          "Expected Improvement",
    "puda":        "PUDA (Proposed)",
    "pred_only":   "Prediction Only",
    "pred_unc":    "Prediction + Uncertainty",
    "pred_div":    "Prediction + Diversity",
    "full":        "Full PUDA",
}


def _initial_indices(pool_size: int, requested: int, seed: int) -> List[int]:
    """Select a reproducible initial seed set shared by strategies for a seed."""
    n_initial = min(requested, pool_size - 1)
    if n_initial < 1:
        raise ValueError("The training pool must contain at least two candidates.")
    return np.random.RandomState(seed).choice(
        pool_size, size=n_initial, replace=False
    ).astype(int).tolist()


def _dataset_fingerprint(dataset_id: str, X, y, candidate_ids: List[str]) -> str:
    """Fingerprint exact benchmark inputs without depending on object reprs."""
    digest = hashlib.sha256()
    digest.update(dataset_id.encode("utf-8"))
    digest.update(json.dumps(list(map(str, X.columns))).encode("utf-8"))
    digest.update(np.ascontiguousarray(X.to_numpy()).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(y, dtype=float)).tobytes())
    digest.update(json.dumps(list(map(str, candidate_ids))).encode("utf-8"))
    return digest.hexdigest()


def _run_single_experiment(
    X_pool, y_pool, pool_ids,
    X_test, y_test, test_ids,
    strategy: str,
    model_type: str,
    initial_samples: int,
    batch_size: int,
    total_budget: int,
    random_seed: int,
    extra_kwargs: Optional[dict] = None,
    seed_indices: Optional[List[int]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> dict:
    """
    Run one complete AL experiment for a single strategy+seed combination.
    Returns curve data and final metrics.
    """
    extra = extra_kwargs or {}
    config = ExperimentConfig(
        strategy=strategy,
        model_type=model_type,
        initial_samples=initial_samples,
        batch_size=batch_size,
        total_budget=total_budget,
        random_seed=random_seed,
        n_estimators=100,   # faster RF for benchmarks
        **extra,
    )
    engine = ActiveLearningEngine(config)
    started = time.time()
    pool_maximum = float(y_pool.max())
    state = engine.initialize(
        X_pool, y_pool, pool_ids, X_test, y_test, test_ids,
        seed_indices=seed_indices,
        dataset_maximum=pool_maximum,
    )

    budget_curve, acquisition_budget_curve = [], []
    rmse_curve, mae_curve, diversity_curve = [], [], []
    iteration_runtime_curve = []
    initial_values = [
        float(state.acquired_values[index]) for index in state.labeled_indices
    ]
    initial_best = max(initial_values)
    initial_ids = [pool_ids[index] for index in state.labeled_indices]
    acquired_in_order: List[float] = []
    acquired_ids: List[str] = []
    previously_labeled = set(state.labeled_indices)
    failure_reason = None
    target_budget = min(total_budget, len(X_pool))

    while len(state.labeled_indices) < target_budget:
        fit_budget = len(state.labeled_indices)
        if fit_budget >= total_budget:
            break
        try:
            result, state = engine.run_iteration(state)
        except RuntimeError as exc:
            failure_reason = str(exc)
            newly_labeled = [
                index for index in state.labeled_indices
                if index not in previously_labeled
            ]
            acquired_in_order.extend(
                float(state.acquired_values[index]) for index in newly_labeled
            )
            acquired_ids.extend(pool_ids[index] for index in newly_labeled)
            previously_labeled.update(newly_labeled)
            break
        except Exception as exc:
            logger.exception(
                "Benchmark iteration failed | strategy=%s seed=%d",
                strategy, random_seed,
            )
            failure_reason = f"{type(exc).__name__}: {exc}"
            newly_labeled = [
                index for index in state.labeled_indices
                if index not in previously_labeled
            ]
            acquired_in_order.extend(
                float(state.acquired_values[index]) for index in newly_labeled
            )
            acquired_ids.extend(pool_ids[index] for index in newly_labeled)
            previously_labeled.update(newly_labeled)
            break

        budget_curve.append(fit_budget)
        acquisition_budget_curve.append(len(state.labeled_indices))
        rmse_curve.append(result.test_rmse)
        mae_curve.append(result.test_mae)
        diversity_curve.append(result.diversity_score)
        iteration_runtime_curve.append(result.iteration_time_seconds)
        acquired_in_order.extend(result.selected_true_values)
        acquired_ids.extend(result.selected_candidate_ids)
        previously_labeled.update(state.labeled_indices)
        if progress_callback is not None:
            progress_callback(len(state.labeled_indices))

    if failure_reason is None and len(state.labeled_indices) < total_budget:
        failure_reason = (
            f"Training pool exhausted at {len(state.labeled_indices)} candidates "
            f"before the requested budget of {total_budget}."
        )

    final_mae = None
    final_rmse = None
    if state.labeled_indices:
        try:
            final_model, final_preprocessor = engine.train_model(state)
            final_metrics = final_model.evaluate(
                final_preprocessor.transform(X_test), y_test.to_numpy()
            )
            final_mae = float(final_metrics["mae"])
            final_rmse = float(final_metrics["rmse"])
        except Exception as exc:
            logger.exception(
                "Final benchmark evaluation failed | strategy=%s seed=%d",
                strategy, random_seed,
            )
            failure_reason = f"Final evaluation {type(exc).__name__}: {exc}"

    discovery_budgets, bf_curve = discovery_curve_with_initial(
        initial_best, acquired_in_order, len(initial_ids),
    )
    regret_c = cumulative_regret(bf_curve, pool_maximum, mode="maximize")
    naudc = normalized_area_under_discovery_curve(
        bf_curve, pool_maximum, mode="maximize", budgets=discovery_budgets,
    )
    top10 = top_k_discovery_rate(
        initial_values + acquired_in_order,
        y_pool.values,
        k=min(10, len(y_pool)),
        mode="maximize",
    )
    target_reached = len(state.labeled_indices) >= total_budget
    status = "completed" if target_reached and failure_reason is None else "incomplete"
    if failure_reason and not budget_curve:
        status = "failed"

    return {
        "strategy": strategy,
        "seed": random_seed,
        "status": status,
        "error": failure_reason,
        "initial_candidate_ids": initial_ids,
        "acquired_candidate_ids": acquired_ids,
        "requested_budget": total_budget,
        "actual_budget": len(state.labeled_indices),
        "actual_acquisitions": len(acquired_in_order),
        "budget_curve": budget_curve,
        "acquisition_budget_curve": acquisition_budget_curve,
        "discovery_budget_curve": discovery_budgets,
        "rmse_curve": rmse_curve,
        "mae_curve": mae_curve,
        "best_found_curve": bf_curve,
        "regret_curve": regret_c,
        "diversity_curve": diversity_curve,
        "iteration_runtime_curve": iteration_runtime_curve,
        "acquired_values_in_order": acquired_in_order,
        "naudc": naudc,
        "top10_discovery_rate": top10,
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        "final_regret": regret_c[-1] if regret_c else pool_maximum - initial_best,
        "global_maximum": pool_maximum,
        "runtime_seconds": time.time() - started,
    }


def run_strategy_benchmark(
    db: Session,
    experiment_id: str,
    strategies: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    model_type: str = "rf",
) -> dict:
    """
    Run a matched multi-strategy benchmark rooted at an existing experiment.

    Uses the experiment's fixed dataset split and budget settings. Each seed
    gets one initial set shared across strategies; results are persisted
    incrementally so failed and incomplete runs remain inspectable.

    Model defaults are fixed for every strategy in this benchmark; optional
    model tuning is deliberately not run here.
    """
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")

    dataset = get_dataset_by_id(db, exp.dataset_id)
    if not dataset or dataset.status != "ready":
        raise ValueError("Benchmark dataset is missing or not ready.")
    X, y, candidate_ids = load_dataset_arrays(dataset)
    if len(X) != len(y) or len(X) != len(candidate_ids) or len(X) < 3:
        raise ValueError("Benchmark dataset arrays must be aligned and contain at least 3 rows.")
    if exp.batch_size < 1 or exp.total_budget <= exp.initial_samples:
        raise ValueError("Benchmark budgets must have a positive batch and exceed initial_samples.")

    split = initial_split(X, y, candidate_ids, test_size=0.20, random_state=exp.random_seed)
    X_pool  = split["pool_X"];  y_pool  = split["pool_y"];  pool_ids  = split["pool_ids"]
    X_test  = split["test_X"];  y_test  = split["test_y"];  test_ids  = split["test_ids"]

    strategies_to_run = list(dict.fromkeys(strategies or STRATEGIES))
    unknown_strategies = sorted(set(strategies_to_run) - set(STRATEGIES))
    if unknown_strategies:
        raise ValueError(f"Unsupported benchmark strategies: {unknown_strategies}")
    if model_type not in {"rf", "gp", "ensemble"}:
        raise ValueError(f"Unsupported benchmark model type: {model_type}")
    seeds_to_run = list(dict.fromkeys(
        DEFAULT_BENCHMARK_SEEDS if seeds is None else seeds
    ))
    if not seeds_to_run or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds_to_run
    ):
        raise ValueError("Benchmark seeds must be a non-empty list of non-negative integers.")

    t0 = time.time()
    all_results: List[dict] = []
    model_configuration = {
        "model_type": model_type,
        "model_params": {},
    }
    if model_type == "rf":
        model_configuration.update({
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": -1,
        })
    elif model_type == "gp":
        model_configuration.update({
            "n_restarts_optimizer": 3,
            "length_scale": 1.0,
            "noise_level": 0.1,
            "normalize_y": True,
            "kernel": "ConstantKernel * RBF + WhiteKernel",
        })
    else:
        model_configuration.update({
            "n_members": 5,
            "hidden_layer_sizes": [128, 64, 32],
            "max_iter": 500,
            "learning_rate_init": 1e-3,
        })
    model_configuration.update({
        "use_pca": False,
        "pca_n_components": 50,
    })
    configuration = {
        "protocol_version": "matched-multiseed-v1",
        "dataset_id": dataset.id,
        "dataset_fingerprint": _dataset_fingerprint(dataset.id, X, y, candidate_ids),
        "target_column": exp.target_column,
        "feature_columns": list(map(str, X.columns)),
        "split": {
            "method": "train_test_split",
            "test_size": 0.20,
            "random_state": exp.random_seed,
            "pool_size": len(X_pool),
            "test_size_rows": len(X_test),
            "pool_candidate_ids": list(map(str, pool_ids)),
            "test_candidate_ids": list(map(str, test_ids)),
        },
        "strategies": strategies_to_run,
        "seeds": seeds_to_run,
        "model": model_configuration,
        "preprocessing": {
            "variance_threshold": 1e-6,
            "correlation_threshold": 0.95,
            "scaler": "StandardScaler",
            "use_pca": False,
        },
        "runtime_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "acquisition_parameters": {
            "xi": exp.xi, "kappa": exp.kappa,
            "alpha": exp.alpha, "beta": exp.beta, "gamma": exp.gamma,
        },
        "initial_samples": exp.initial_samples,
        "batch_size": exp.batch_size,
        "total_budget": exp.total_budget,
        "base_seed": exp.random_seed,
        "test_targets_used_for_selection": False,
    }
    benchmark_record = BenchmarkRun(
        id=str(uuid.uuid4()),
        experiment_id=experiment_id,
        status="running",
        model_type=model_type,
        configuration=configuration,
        results=[],
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
    )
    db.add(benchmark_record)
    db.commit()

    for strategy in strategies_to_run:
        for seed in seeds_to_run:
            logger.info("Benchmark: strategy=%s seed=%d", strategy, seed)
            run_started = time.time()
            initial_indices = []
            try:
                initial_indices = _initial_indices(
                    len(X_pool), exp.initial_samples, seed
                )
                active_run = {
                    "strategy": strategy,
                    "seed": seed,
                    "initial_candidate_ids": [
                        str(pool_ids[index]) for index in initial_indices
                    ],
                    "requested_budget": exp.total_budget,
                    "actual_budget": len(initial_indices),
                }
                configuration = {
                    **benchmark_record.configuration,
                    "heartbeat_at": datetime.utcnow().isoformat(),
                    "active_run": active_run,
                }
                benchmark_record.configuration = configuration
                db.commit()

                def save_progress(actual_budget: int) -> None:
                    progress = {
                        **active_run,
                        "actual_budget": actual_budget,
                    }
                    benchmark_record.configuration = {
                        **benchmark_record.configuration,
                        "heartbeat_at": datetime.utcnow().isoformat(),
                        "active_run": progress,
                    }
                    db.commit()

                with _ACTIVE_BENCHMARK_RUN_IDS_LOCK:
                    _ACTIVE_BENCHMARK_RUN_IDS.add(benchmark_record.id)
                try:
                    result = _run_single_experiment(
                        X_pool, y_pool, pool_ids, X_test, y_test, test_ids,
                        strategy=strategy, model_type=model_type,
                        initial_samples=exp.initial_samples,
                        batch_size=exp.batch_size,
                        total_budget=exp.total_budget,
                        random_seed=seed,
                        extra_kwargs={
                            "xi": exp.xi, "kappa": exp.kappa,
                            "alpha": exp.alpha, "beta": exp.beta,
                            "gamma": exp.gamma,
                        },
                        seed_indices=initial_indices,
                        progress_callback=save_progress,
                    )
                finally:
                    with _ACTIVE_BENCHMARK_RUN_IDS_LOCK:
                        _ACTIVE_BENCHMARK_RUN_IDS.discard(benchmark_record.id)
            except Exception as exc:
                logger.exception(
                    "Benchmark run failed | strategy=%s seed=%d", strategy, seed
                )
                result = {
                    "strategy": strategy,
                    "seed": seed,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "initial_candidate_ids": [
                        str(pool_ids[index]) for index in initial_indices
                    ],
                    "acquired_candidate_ids": [],
                    "requested_budget": exp.total_budget,
                    "actual_budget": len(initial_indices),
                    "actual_acquisitions": 0,
                    "budget_curve": [],
                    "acquisition_budget_curve": [],
                    "discovery_budget_curve": [],
                    "rmse_curve": [],
                    "mae_curve": [],
                    "best_found_curve": [],
                    "regret_curve": [],
                    "diversity_curve": [],
                    "iteration_runtime_curve": [],
                    "acquired_values_in_order": [],
                    "naudc": None,
                    "top10_discovery_rate": None,
                    "final_mae": None,
                    "final_rmse": None,
                    "final_regret": None,
                    "global_maximum": float(y_pool.max()),
                    "runtime_seconds": time.time() - run_started,
                }
            all_results.append(result)
            benchmark_record.results = list(all_results)
            benchmark_record.configuration = {
                **benchmark_record.configuration,
                "heartbeat_at": datetime.utcnow().isoformat(),
            }
            benchmark_record.configuration.pop("active_run", None)
            db.commit()

    elapsed = time.time() - t0
    summary = _build_summary(
        all_results, seeds_to_run, expected_strategies=strategies_to_run,
    )
    statuses = [result["status"] for result in all_results]
    if statuses and all(status == "completed" for status in statuses):
        benchmark_status = "completed"
    elif any(status in ("completed", "incomplete") for status in statuses):
        benchmark_status = "partial"
    else:
        benchmark_status = "failed"
    summary["experiment_id"] = experiment_id
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["n_seeds"] = len(seeds_to_run)
    summary["seeds"] = seeds_to_run
    summary["model_type"] = model_type
    summary["strategies_run"] = strategies_to_run
    summary["benchmark_id"] = benchmark_record.id
    summary["benchmark_status"] = benchmark_status
    summary["failed_or_incomplete_runs"] = [
        {
            "strategy": result["strategy"],
            "seed": result["seed"],
            "status": result["status"],
            "error": result.get("error"),
            "actual_budget": result["actual_budget"],
        }
        for result in all_results if result["status"] != "completed"
    ]
    summary["per_seed_results"] = all_results
    summary["configuration"] = configuration

    logger.info(
        "Benchmark done | %d strategies × %d seeds = %d runs | %.1fs",
        len(strategies_to_run), len(seeds_to_run), len(all_results), elapsed,
    )
    benchmark_record.status = benchmark_status
    benchmark_record.summary = summary
    benchmark_record.elapsed_seconds = elapsed
    benchmark_record.finished_at = datetime.utcnow()
    benchmark_record.error_summary = "; ".join(
        f"{run['strategy']} seed {run['seed']}: {run.get('error') or run['status']}"
        for run in all_results if run["status"] != "completed"
    ) or None
    db.commit()
    return summary


def run_ablation_study(
    db: Session,
    experiment_id: str,
    seeds: Optional[List[int]] = None,
    model_type: str = "rf",
) -> dict:
    """
    Run a PUDA ablation study: compare 4 PUDA variants under matched conditions.

    Variants (from AblationPUDA):
      pred_only  — α=1.0, β=0.0, γ=0.0  (prediction only)
      pred_unc   — α=0.5, β=0.5, γ=0.0  (prediction + uncertainty)
      pred_div   — α=0.5, β=0.0, γ=0.5  (prediction + diversity)
      full       — α=0.4, β=0.35, γ=0.25 (full PUDA)

    All variants use strategy="puda" with weights overridden.
    """
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")

    dataset = get_dataset_by_id(db, exp.dataset_id)
    X, y, candidate_ids = load_dataset_arrays(dataset)
    split = initial_split(X, y, candidate_ids, test_size=0.20, random_state=exp.random_seed)
    X_pool  = split["pool_X"];  y_pool  = split["pool_y"];  pool_ids  = split["pool_ids"]
    X_test  = split["test_X"];  y_test  = split["test_y"];  test_ids  = split["test_ids"]

    seeds_to_run = seeds or [exp.random_seed]
    t0 = time.time()
    all_results: List[dict] = []

    for mode_name, weights in ABLATION_CONFIGS.items():
        for seed in seeds_to_run:
            logger.info("Ablation: mode=%s seed=%d", mode_name, seed)
            result = _run_single_experiment(
                X_pool, y_pool, pool_ids, X_test, y_test, test_ids,
                strategy="puda", model_type=model_type,
                initial_samples=exp.initial_samples,
                batch_size=exp.batch_size,
                total_budget=exp.total_budget,
                random_seed=seed,
                extra_kwargs=weights,
            )
            result["strategy"] = mode_name  # relabel for display
            all_results.append(result)

    elapsed = time.time() - t0
    summary = _build_summary(all_results, seeds_to_run, label_prefix="ablation")
    summary.update({
        "experiment_id": experiment_id,
        "elapsed_seconds": round(elapsed, 1),
        "n_seeds": len(seeds_to_run),
        "model_type": model_type,
        "ablation_variants": ABLATION_MODES,
        "formulas": {
            "pred_only": "Score = α·norm(μ)  [α=1.0, β=0, γ=0]",
            "pred_unc":  "Score = α·norm(μ) + β·norm(σ)  [α=β=0.5, γ=0]",
            "pred_div":  "Score = α·norm(μ) + γ·div(x,S)  [α=γ=0.5, β=0]",
            "full":      "Score = α·norm(μ) + β·norm(σ) + γ·div(x,S)  [α=0.4, β=0.35, γ=0.25]",
        },
        "interpretation": (
            "If 'full' PUDA outperforms all ablation variants, each component contributes. "
            "If 'pred_unc' matches 'full', the diversity term may not be necessary for this dataset. "
            "Conclusions depend on the actual experimental results shown above."
        ),
    })
    return summary


def recover_interrupted_benchmarks(
    db: Session,
    *,
    now: Optional[datetime] = None,
    stale_after: timedelta = INTERRUPTED_BENCHMARK_TIMEOUT,
) -> int:
    """Mark stale running benchmark records without rerunning or discarding work."""
    recovery_time = now or datetime.utcnow()
    if recovery_time.tzinfo is not None:
        recovery_time = recovery_time.astimezone(timezone.utc).replace(tzinfo=None)

    with _ACTIVE_BENCHMARK_RUN_IDS_LOCK:
        active_ids = set(_ACTIVE_BENCHMARK_RUN_IDS)

    recovered_count = 0
    running_records = db.query(BenchmarkRun).filter(
        BenchmarkRun.status == "running"
    ).all()
    for record in running_records:
        if record.id in active_ids:
            continue

        configuration = dict(record.configuration or {})
        heartbeat_value = configuration.get("heartbeat_at")
        heartbeat = None
        if isinstance(heartbeat_value, str):
            try:
                heartbeat = datetime.fromisoformat(heartbeat_value)
            except ValueError:
                logger.warning(
                    "Invalid benchmark heartbeat for %s; falling back to started_at",
                    record.id,
                )
        heartbeat = heartbeat or record.started_at or record.created_at
        if heartbeat is None:
            logger.warning(
                "Cannot recover benchmark %s without a start or heartbeat timestamp",
                record.id,
            )
            continue
        if heartbeat.tzinfo is not None:
            heartbeat = heartbeat.astimezone(timezone.utc).replace(tzinfo=None)
        if recovery_time - heartbeat < stale_after:
            continue

        interrupted_message = (
            "Interrupted: no heartbeat for at least "
            f"{stale_after.total_seconds() / 3600:g} hours. "
            "The benchmark was not rerun."
        )
        results = list(record.results or [])
        active_run = configuration.get("active_run")
        if isinstance(active_run, dict):
            interrupted_result = {
                **active_run,
                "status": "failed",
                "error": interrupted_message,
                "actual_acquisitions": 0,
                "budget_curve": [],
                "acquisition_budget_curve": [],
                "discovery_budget_curve": [],
                "rmse_curve": [],
                "mae_curve": [],
                "best_found_curve": [],
                "regret_curve": [],
                "diversity_curve": [],
                "iteration_runtime_curve": [],
                "acquired_values_in_order": [],
                "acquired_candidate_ids": [],
                "naudc": None,
                "top10_discovery_rate": None,
                "final_mae": None,
                "final_rmse": None,
                "final_regret": None,
                "global_maximum": None,
                "runtime_seconds": None,
            }
            results.append(interrupted_result)
            record.results = results

        completed_or_partial_results = [
            result for result in results
            if result.get("status") in ("completed", "incomplete")
        ]
        record.status = "partial" if completed_or_partial_results else "failed"
        record.error_summary = (
            f"{record.error_summary}; {interrupted_message}"
            if record.error_summary
            else interrupted_message
        )
        record.finished_at = recovery_time
        if record.started_at is not None:
            record.elapsed_seconds = max(
                0.0, (recovery_time - record.started_at).total_seconds()
            )
        configuration["recovered_at"] = recovery_time.isoformat()
        record.configuration = configuration

        summary = dict(record.summary or {})
        summary["benchmark_status"] = record.status
        summary["interrupted"] = True
        summary["recovery_message"] = interrupted_message
        summary["failed_or_incomplete_runs"] = [
            {
                "strategy": result.get("strategy"),
                "seed": result.get("seed"),
                "status": result.get("status"),
                "error": result.get("error"),
                "actual_budget": result.get("actual_budget"),
            }
            for result in results
            if result.get("status") != "completed"
        ]
        summary["run_status_counts"] = {
            status: sum(result.get("status") == status for result in results)
            for status in ("completed", "incomplete", "failed")
        }
        if "per_seed_results" in summary:
            summary["per_seed_results"] = results
        record.summary = summary
        db.commit()
        recovered_count += 1

    return recovered_count


def get_benchmark_run(db: Session, experiment_id: str, benchmark_id: str) -> dict:
    """Read a persisted benchmark snapshot without changing its parent experiment."""
    record = (
        db.query(BenchmarkRun)
        .filter(
            BenchmarkRun.id == benchmark_id,
            BenchmarkRun.experiment_id == experiment_id,
        )
        .first()
    )
    if record is None:
        raise ValueError(f"Benchmark run {benchmark_id} not found.")
    return {
        "benchmark_id": record.id,
        "experiment_id": record.experiment_id,
        "status": record.status,
        "model_type": record.model_type,
        "configuration": record.configuration,
        "results": record.results,
        "summary": record.summary,
        "error_summary": record.error_summary,
        "elapsed_seconds": record.elapsed_seconds,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "started_at": record.started_at.isoformat() if record.started_at else None,
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
    }


def get_calibration_metrics(
    db: Session,
    experiment_id: str,
) -> dict:
    """
    Compute pre-acquisition calibration metrics on acquired candidates.

    CandidatePredictionRecord rows are emitted from the pool predictions made
    before selection/acquisition in that iteration. For each candidate, use
    only predictions from this experiment at or before its first acquisition
    iteration, selecting the latest eligible prediction.

    Only ExperimentalResult values are read as targets. Unobserved pool values
    are never accessed by this calibration path.
    """
    from backend.app.models.result import CandidatePredictionRecord, ExperimentalResult

    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")

    # Targets are limited to observations already revealed by acquisition.
    acq = db.query(ExperimentalResult).filter(
        ExperimentalResult.experiment_id == experiment_id
    ).order_by(
        ExperimentalResult.iteration_num,
        ExperimentalResult.acquired_at,
        ExperimentalResult.id,
    ).all()
    if not acq:
        return {
            "error": "No acquired observations are available for calibration.",
            "n_aligned": 0, "n_excluded": 0, "exclusion_counts": {},
            "minimum_required": 5,
        }

    preds = (
        db.query(CandidatePredictionRecord)
        .filter(
            CandidatePredictionRecord.experiment_id == experiment_id,
        )
        .all()
    )
    preds_by_candidate: dict[str, list] = {}
    for prediction in preds:
        preds_by_candidate.setdefault(prediction.candidate_id, []).append(prediction)

    # A candidate should be acquired once. If legacy data has duplicates, use
    # its first reveal so later observations cannot change its calibration pair.
    first_acquisition = {}
    duplicate_acquisitions = 0
    for observation in acq:
        if observation.candidate_id in first_acquisition:
            duplicate_acquisitions += 1
        else:
            first_acquisition[observation.candidate_id] = observation

    y_true, y_pred_mean, y_pred_std = [], [], []
    exclusions: dict[str, int] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for cid, observation in first_acquisition.items():
        candidate_preds = preds_by_candidate.get(cid, [])
        eligible = [p for p in candidate_preds if p.iteration_num <= observation.iteration_num]
        if not eligible:
            exclude("prediction_after_acquisition" if candidate_preds else "no_prediction_record")
            continue

        # Latest pre-acquisition prediction; ID makes duplicate same-iteration
        # records deterministic without relying on absent prediction timestamps.
        prediction = max(eligible, key=lambda p: (p.iteration_num, p.id or ""))
        true_value = observation.true_value
        mean = prediction.predicted_mean
        std = prediction.predicted_std
        if true_value is None or not np.isfinite(true_value):
            exclude("invalid_true_value")
        elif mean is None or not np.isfinite(mean):
            exclude("missing_or_invalid_predicted_mean")
        elif std is None:
            exclude("missing_predicted_std")
        elif not np.isfinite(std):
            exclude("invalid_predicted_std")
        elif std <= 0:
            exclude("non_positive_predicted_std")
        else:
            y_true.append(float(true_value))
            y_pred_mean.append(float(mean))
            y_pred_std.append(float(std))

    n_valid = len(y_true)
    n_excluded = sum(exclusions.values()) + duplicate_acquisitions
    exclusion_counts = dict(exclusions)
    if duplicate_acquisitions:
        exclusion_counts["duplicate_acquisition_record"] = duplicate_acquisitions

    if n_valid < 5:
        return {
            "error": f"Insufficient calibration data: {n_valid} valid pre-acquisition pairs; at least 5 are required.",
            "n_aligned": n_valid,
            "n_observations": n_valid,
            "n_excluded": n_excluded,
            "exclusion_counts": exclusion_counts,
            "minimum_required": 5,
        }

    y_true = np.asarray(y_true, dtype=float)
    y_pred_mean = np.asarray(y_pred_mean, dtype=float)
    y_pred_std = np.asarray(y_pred_std, dtype=float)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = float(mean_absolute_error(y_true, y_pred_mean))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_mean)))
    cal = expected_calibration_error(y_true, y_pred_mean, y_pred_std, n_bins=10)

    return {
        "experiment_id": experiment_id,
        "iteration": exp.current_iteration,
        "n_aligned": n_valid,
        "n_observations": cal["n_observations"],
        "n_excluded": n_excluded,
        "exclusion_counts": exclusion_counts,
        "minimum_required": 5,
        "mae_on_observed": mae,
        "rmse_on_observed": rmse,
        "ece": cal["ece"],
        "coverage_90pct": cal["coverage_90pct"],
        "coverage_95pct": cal["coverage_95pct"],
        "reliability_diagram": cal["reliability_diagram"],
        "interpretation": (
            "Coverage-based empirical calibration error is the mean absolute gap between nominal "
            "and observed Gaussian interval coverage over the reliability levels. It is a coverage "
            "diagnostic, not a complete test of probabilistic calibration."
        ),
        "caveat": (
            "Pairs use the latest prediction made no later than each candidate's first acquisition. "
            "The sample is selected by the acquisition policy and is not an independent test set; "
            "unobserved targets are not read."
        ),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_summary(
    results: List[dict],
    seeds: List[int],
    label_prefix: str = "strategy",
    expected_strategies: Optional[List[str]] = None,
) -> dict:
    """
    Aggregate per-(strategy, seed) results into chart-ready summary.
    Aggregate only values observed at each exact candidate-budget checkpoint.
    Runtime summaries use completed runs only.
    A single seed has an undefined sample standard deviation, represented by
    None rather than a misleading zero.
    """
    from collections import defaultdict

    by_strategy: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)
    strategy_order = expected_strategies or list(by_strategy)

    strategy_summaries = []
    discovery_series = []
    rmse_series = []
    mae_series = []
    regret_series = []
    diversity_series = []

    for strat in strategy_order:
        runs = by_strategy.get(strat, [])
        successful_runs = [
            run for run in runs
            if run.get("status", "completed") == "completed"
        ]

        # Mean curves across seeds
        def aggregate_curve(value_key: str, budget_key: str):
            by_budget: Dict[int, List[float]] = {}
            for run in successful_runs:
                for budget, value in zip(run[budget_key], run[value_key]):
                    if value is None:
                        continue
                    by_budget.setdefault(int(budget), []).append(float(value))
            axis = sorted(by_budget)
            means = [float(np.mean(by_budget[b])) for b in axis]
            stds = [
                float(np.std(by_budget[b], ddof=1)) if len(by_budget[b]) > 1 else None
                for b in axis
            ]
            counts = [len(by_budget[b]) for b in axis]
            return axis, means, stds, counts

        # Each curve is aggregated on its own actual cumulative-candidate checkpoints.
        discovery_budget_axis, mean_best, std_best, n_best = aggregate_curve(
            "best_found_curve", "discovery_budget_curve",
        )
        _, mean_regret, std_regret, n_regret = aggregate_curve(
            "regret_curve", "discovery_budget_curve"
        )
        iteration_budget_axis, mean_rmse, std_rmse, n_rmse = aggregate_curve(
            "rmse_curve", "budget_curve"
        )
        _, mean_mae, std_mae, n_mae = aggregate_curve("mae_curve", "budget_curve")
        diversity_budget_axis, mean_div, std_div, n_div = aggregate_curve(
            "diversity_curve", "acquisition_budget_curve"
        )

        def values_for(key: str) -> List[float]:
            return [
                float(run[key]) for run in successful_runs
                if run.get(key) is not None
            ]

        def mean_and_sample_std(values: List[float]) -> tuple[Optional[float], Optional[float]]:
            if not values:
                return None, None
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else None
            return mean, std

        naudc_mean, naudc_std = mean_and_sample_std(values_for("naudc"))
        top10_mean, top10_std = mean_and_sample_std(
            values_for("top10_discovery_rate")
        )
        rmse_mean, rmse_std = mean_and_sample_std(values_for("final_rmse"))
        mae_mean, mae_std = mean_and_sample_std(values_for("final_mae"))
        regret_mean, regret_std = mean_and_sample_std(values_for("final_regret"))
        budget_mean, budget_std = mean_and_sample_std(
            [float(run["actual_budget"]) for run in successful_runs]
        )
        runtime_mean, runtime_std = mean_and_sample_std(
            [
                float(run["runtime_seconds"]) for run in successful_runs
                if run.get("runtime_seconds") is not None
            ]
        )

        color = STRATEGY_COLORS.get(strat, "#888")
        label = STRATEGY_LABELS.get(strat, strat)

        series_base = {"strategy": strat, "label": label, "color": color}

        discovery_series.append({
            **series_base,
            "budget": discovery_budget_axis,
            "budget_unit": "cumulative_candidates",
            "best_found": mean_best,
            "best_found_std": std_best,
            "n_runs": n_best,
        })
        rmse_series.append({
            **series_base, "budget": iteration_budget_axis,
            "budget_unit": "training_candidates", "rmse": mean_rmse,
            "rmse_std": std_rmse, "n_runs": n_rmse,
        })
        mae_series.append({
            **series_base, "budget": iteration_budget_axis,
            "budget_unit": "training_candidates", "mae": mean_mae,
            "mae_std": std_mae, "n_runs": n_mae,
        })
        regret_series.append({
            **series_base, "budget": discovery_budget_axis,
            "budget_unit": "cumulative_candidates", "regret": mean_regret,
            "regret_std": std_regret, "n_runs": n_regret,
        })
        diversity_series.append({
            **series_base, "budget": diversity_budget_axis,
            "budget_unit": "cumulative_candidates", "diversity": mean_div,
            "diversity_std": std_div, "n_runs": n_div,
        })

        strategy_summaries.append({
            "strategy": strat,
            "label": label,
            "color": color,
            "n_seeds": len(successful_runs),
            "n_requested_seeds": len(seeds),
            "n_incomplete": sum(run["status"] == "incomplete" for run in runs),
            "n_failed": sum(run["status"] == "failed" for run in runs),
            "naudc_mean": naudc_mean,
            "naudc_std": naudc_std,
            "top10_mean": top10_mean,
            "top10_std": top10_std,
            "final_mae_mean": mae_mean,
            "final_mae_std": mae_std,
            "final_rmse_mean": rmse_mean,
            "final_rmse_std": rmse_std,
            "final_regret_mean": regret_mean,
            "final_regret_std": regret_std,
            "actual_budget_mean": budget_mean,
            "actual_budget_std": budget_std,
            "runtime_seconds_mean": runtime_mean,
            "runtime_seconds_std": runtime_std,
            "global_maximum": runs[0].get("global_maximum") if runs else None,
        })

    # Rank only strategies with completed results; never infer a winner from
    # missing runs or failed summary values.
    strategy_summaries.sort(
        key=lambda row: (
            row["naudc_mean"] is None,
            -(row["naudc_mean"] or 0.0),
            strategy_order.index(row["strategy"]),
        )
    )
    rank = 0
    for row in strategy_summaries:
        if row["naudc_mean"] is None:
            row["rank"] = None
        else:
            rank += 1
            row["rank"] = rank

    return {
        "summary_table": strategy_summaries,
        "discovery_curve": discovery_series,
        "rmse_curve": rmse_series,
        "mae_curve": mae_series,
        "regret_curve": regret_series,
        "diversity_curve": diversity_series,
        "best_strategy": next(
            (row["strategy"] for row in strategy_summaries if row["rank"] == 1),
            None,
        ),
        "global_maximum": results[0].get("global_maximum") if results else None,
    }
