"""
Experiment service: create, initialize, run iterations, persist state.

Key change from original: ExperimentState is now serialized to disk at
state_path after every iteration so the engine can be restored after a
server restart without losing labeled-set history or random-seed continuity.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from backend.app.models.experiment import Experiment, ExperimentIteration, TuningRun
from backend.app.models.result import CandidatePredictionRecord, ExperimentalResult
from backend.app.schemas.experiment import ExperimentCreate
from backend.app.services.dataset_service import get_dataset_by_id, load_dataset_arrays
from backend.config import get_settings
from backend.ml.active_learning.engine import (
    ActiveLearningEngine,
    ExperimentConfig,
    ExperimentState,
    IterationResult,
)
from backend.ml.data.preprocessing import initial_split

logger = logging.getLogger(__name__)
settings = get_settings()

# ── In-memory fast-path cache ─────────────────────────────────────────────────
# Falls back to disk if process-level cache is cold (e.g. after restart).
_ENGINE_CACHE: dict = {}
_STATE_CACHE: dict = {}

_STATE_DIR = Path(settings.model_dir) / "states"
PREDICTION_POOL_PERSIST_LIMIT = 500


def _predictions_to_persist(all_predictions, selected_candidate_ids):
    """Keep the bounded pool sample and every acquired candidate prediction."""
    selected_ids = set(selected_candidate_ids)
    selected = []
    seen = set()

    for prediction in all_predictions[:PREDICTION_POOL_PERSIST_LIMIT]:
        selected.append(prediction)
        seen.add(prediction.candidate_id)

    for prediction in all_predictions[PREDICTION_POOL_PERSIST_LIMIT:]:
        if prediction.candidate_id in selected_ids and prediction.candidate_id not in seen:
            selected.append(prediction)
            seen.add(prediction.candidate_id)

    # Selected candidates should normally be in all_predictions. Fail loudly if
    # engine output ever violates that invariant instead of silently losing a pair.
    missing_selected = selected_ids - seen
    if missing_selected:
        raise ValueError(
            "Selected candidates have no pre-acquisition prediction: "
            + ", ".join(sorted(missing_selected))
        )
    return selected


def _state_path(experiment_id: str) -> Path:
    """Disk path for serialized ExperimentState."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR / f"{experiment_id}.pkl"


def _save_state(experiment_id: str, state: ExperimentState) -> None:
    p = _state_path(experiment_id)
    with open(p, "wb") as f:
        pickle.dump(state, f)
    logger.debug("State persisted to %s", p)


def _load_state(experiment_id: str) -> Optional[ExperimentState]:
    p = _state_path(experiment_id)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Could not load state from %s: %s", p, exc)
        return None


# ── Core CRUD ─────────────────────────────────────────────────────────────────

def list_experiments(db: Session, skip: int = 0, limit: int = 50) -> list:
    """Return experiments ordered by most-recently updated."""
    experiments = (
        db.query(Experiment)
        .order_by(Experiment.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    changed = False
    for exp in experiments:
        maximum = _dataset_maximum(db, exp)
        if exp.global_maximum != maximum:
            exp.global_maximum = maximum
            changed = True
    if changed:
        db.commit()
    return experiments


def create_experiment(db: Session, req: ExperimentCreate) -> Experiment:
    dataset = get_dataset_by_id(db, req.dataset_id)
    if not dataset:
        raise ValueError(f"Dataset {req.dataset_id} not found.")
    if dataset.status != "ready":
        raise ValueError(f"Dataset is not ready (status={dataset.status}).")

    exp = Experiment(
        id=str(uuid.uuid4()),
        dataset_id=req.dataset_id,
        name=req.name or f"{req.strategy.upper()} experiment",
        strategy=req.strategy,
        model_type=req.model_type,
        initial_samples=req.initial_samples,
        batch_size=req.batch_size,
        total_budget=req.total_budget,
        random_seed=req.random_seed,
        target_column=req.target_column,
        xi=req.xi, kappa=req.kappa,
        alpha=req.alpha, beta=req.beta, gamma=req.gamma,
        use_pca=req.use_pca, pca_n_components=req.pca_n_components,
        status="created",
        created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    logger.info("Experiment created: %s (strategy=%s)", exp.id, exp.strategy)
    return exp


def initialize_experiment(db: Session, experiment_id: str) -> dict:
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")
    if exp.status not in ("created", "paused"):
        raise ValueError(f"Cannot initialize (status={exp.status}).")

    dataset = get_dataset_by_id(db, exp.dataset_id)
    X, y, candidate_ids = load_dataset_arrays(dataset)

    split = initial_split(X, y, candidate_ids, test_size=0.20, random_state=exp.random_seed)

    config = _exp_to_config(db, exp)
    engine = ActiveLearningEngine(config)
    state = engine.initialize(
        pool_X=split["pool_X"], pool_y=split["pool_y"], pool_ids=split["pool_ids"],
        test_X=split["test_X"], test_y=split["test_y"], test_ids=split["test_ids"],
        dataset_maximum=float(dataset.global_maximum) if dataset.global_maximum is not None else float(y.max()),
    )

    _ENGINE_CACHE[experiment_id] = engine
    _STATE_CACHE[experiment_id] = state
    _save_state(experiment_id, state)          # ← persist to disk

    exp.status = "initialized"
    exp.current_budget_used = len(state.labeled_indices)
    exp.global_maximum = float(state.global_maximum)
    exp.state_path = str(_state_path(experiment_id))
    exp.updated_at = datetime.utcnow()
    db.commit()

    logger.info(
        "Experiment %s initialized | seed=%d | pool=%d | test=%d | max=%.4f",
        experiment_id, len(state.labeled_indices),
        len(state.pool_X), len(state.test_X), state.global_maximum,
    )
    return {
        "experiment_id": experiment_id, "status": "initialized",
        "seed_samples": len(state.labeled_indices),
        "pool_size": len(state.pool_X), "test_size": len(state.test_X),
        "global_maximum": state.global_maximum,
        "training_pool_maximum": state.training_pool_maximum,
    }


def run_iteration(db: Session, experiment_id: str) -> IterationResult:
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")
    if exp.status not in ("initialized", "running"):
        raise ValueError(f"Cannot run iteration (status={exp.status}). Initialize first.")

    engine, state = _get_or_restore_engine_state(db, exp)

    exp.status = "running"
    exp.updated_at = datetime.utcnow()
    db.commit()

    result, state = engine.run_iteration(state)

    _STATE_CACHE[experiment_id] = state
    _save_state(experiment_id, state)          # ← persist to disk

    # Persist iteration record
    iter_record = ExperimentIteration(
        id=str(uuid.uuid4()), experiment_id=experiment_id,
        iteration_num=result.iteration,
        test_mae=result.test_mae, test_rmse=result.test_rmse, test_r2=result.test_r2,
        best_found_value=result.best_found_value, best_found_id=result.best_found_id,
        regret=result.regret, diversity_score=result.diversity_score,
        budget_used=result.budget_used, budget_remaining=result.budget_remaining,
        selected_candidate_ids=result.selected_candidate_ids,
        selected_true_values=result.selected_true_values,
        iteration_time_seconds=result.iteration_time_seconds,
        timestamp=datetime.utcnow(),
    )
    db.add(iter_record)

    for cid, val in zip(result.selected_candidate_ids, result.selected_true_values):
        db.add(ExperimentalResult(
            id=str(uuid.uuid4()), experiment_id=experiment_id,
            iteration_num=result.iteration, candidate_id=cid, true_value=val,
        ))

    predictions_to_persist = _predictions_to_persist(
        result.all_predictions, result.selected_candidate_ids
    )
    for pred in predictions_to_persist:
        db.add(CandidatePredictionRecord(
            id=str(uuid.uuid4()), experiment_id=experiment_id,
            iteration_num=result.iteration,
            candidate_id=pred.candidate_id, pool_index=pred.pool_index,
            predicted_mean=pred.predicted_mean, predicted_std=pred.predicted_std,
            lower_cb=pred.lower_cb, upper_cb=pred.upper_cb,
            acquisition_score=pred.acquisition_score,
            is_selected=pred.is_selected, selection_rank=pred.selection_rank,
            breakdown=pred.breakdown,
        ))

    exp.current_iteration = result.iteration
    exp.current_budget_used = result.budget_used
    exp.best_found_value = result.best_found_value
    exp.best_found_id = result.best_found_id
    exp.status = "completed" if result.budget_remaining <= 0 else "running"
    exp.updated_at = datetime.utcnow()
    db.commit()

    logger.info(
        "Iteration %d done | exp=%s | RMSE=%.4f | best=%.4f",
        result.iteration, experiment_id, result.test_rmse, result.best_found_value,
    )
    return result


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_experiment_status(db: Session, experiment_id: str) -> dict:
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")
    last = (
        db.query(ExperimentIteration)
        .filter(ExperimentIteration.experiment_id == experiment_id)
        .order_by(ExperimentIteration.iteration_num.desc())
        .first()
    )
    reference_maximum = _dataset_maximum(db, exp)
    return {
        "experiment_id": experiment_id, "status": exp.status,
        "current_iteration": exp.current_iteration,
        "budget_used": exp.current_budget_used,
        "budget_remaining": exp.total_budget - exp.current_budget_used,
        "budget_total": exp.total_budget,
        "budget_pct": round(100 * exp.current_budget_used / exp.total_budget, 1),
        "best_found_value": exp.best_found_value, "best_found_id": exp.best_found_id,
        "global_maximum": reference_maximum,
        "latest_rmse": last.test_rmse if last else None,
        "latest_r2": last.test_r2 if last else None,
        "latest_regret": reference_maximum - last.best_found_value
        if last and last.best_found_value is not None else None,
    }


def get_iteration_history(db: Session, experiment_id: str) -> list:
    history = (
        db.query(ExperimentIteration)
        .filter(ExperimentIteration.experiment_id == experiment_id)
        .order_by(ExperimentIteration.iteration_num)
        .all()
    )
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if exp and history:
        reference_maximum = _dataset_maximum(db, exp)
        changed = False
        for iteration in history:
            if iteration.best_found_value is not None:
                corrected_regret = reference_maximum - iteration.best_found_value
                if iteration.regret != corrected_regret:
                    iteration.regret = corrected_regret
                    changed = True
        if exp.global_maximum != reference_maximum:
            exp.global_maximum = reference_maximum
            changed = True
        if changed:
            db.commit()
    return history


def get_candidates_for_iteration(
    db: Session, experiment_id: str,
    iteration: Optional[int] = None,
    page: int = 1, page_size: int = 50,
) -> Tuple[list, int]:
    query = db.query(CandidatePredictionRecord).filter(
        CandidatePredictionRecord.experiment_id == experiment_id
    )
    if iteration is not None:
        query = query.filter(CandidatePredictionRecord.iteration_num == iteration)
    else:
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            query = query.filter(
                CandidatePredictionRecord.iteration_num == exp.current_iteration
            )
    total = query.count()
    records = (
        query.order_by(CandidatePredictionRecord.acquisition_score.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return records, total


def get_candidate_detail(
    db: Session, experiment_id: str, candidate_id: str,
) -> Optional[CandidatePredictionRecord]:
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        return None
    return (
        db.query(CandidatePredictionRecord)
        .filter(
            CandidatePredictionRecord.experiment_id == experiment_id,
            CandidatePredictionRecord.candidate_id == candidate_id,
            CandidatePredictionRecord.iteration_num == exp.current_iteration,
        )
        .first()
    )


def get_discovery_list(db: Session, experiment_id: str, top_n: int = 50) -> list:
    results = (
        db.query(ExperimentalResult)
        .filter(ExperimentalResult.experiment_id == experiment_id)
        .all()
    )
    if not results:
        return []
    seen: dict = {}
    for r in results:
        if r.candidate_id not in seen:
            seen[r.candidate_id] = r
    items = sorted(seen.values(), key=lambda r: r.true_value, reverse=True)[:top_n]
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    prediction_records = db.query(CandidatePredictionRecord).filter(
        CandidatePredictionRecord.experiment_id == experiment_id,
    ).all()
    prediction_map = {(p.iteration_num, p.candidate_id): p for p in prediction_records}
    iteration_records = db.query(ExperimentIteration).filter(
        ExperimentIteration.experiment_id == experiment_id,
    ).all()
    budget_by_iteration = {it.iteration_num: it.budget_used for it in iteration_records}
    return [
        {
            "rank": i + 1,
            "candidate_id": r.candidate_id,
            "true_value": r.true_value,
            "iteration_discovered": r.iteration_num,
            "strategy": exp.strategy if exp else "unknown",
            "budget_at_discovery": budget_by_iteration.get(r.iteration_num),
            "predicted_mean": prediction_map.get((r.iteration_num, r.candidate_id)).predicted_mean
                if prediction_map.get((r.iteration_num, r.candidate_id)) else None,
            "predicted_std": prediction_map.get((r.iteration_num, r.candidate_id)).predicted_std
                if prediction_map.get((r.iteration_num, r.candidate_id)) else None,
            "acquisition_score": prediction_map.get((r.iteration_num, r.candidate_id)).acquisition_score
                if prediction_map.get((r.iteration_num, r.candidate_id)) else None,
            "feature_preview": {},
        }
        for i, r in enumerate(items)
    ]


# ── Embedding endpoint ────────────────────────────────────────────────────────

def get_experiment_embedding(
    db: Session,
    experiment_id: str,
    n_components: int = 2,
) -> dict:
    """
    Compute PCA embedding of the pool's feature matrix.

    Returns a list of candidate records with 2D/3D coordinates,
    model predictions (if available from the latest iteration),
    and observation status.

    PCA is fitted on ALL pool features to give a stable global
    coordinate system independent of the current labeled set.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")

    dataset = get_dataset_by_id(db, exp.dataset_id)
    X, y, candidate_ids = load_dataset_arrays(dataset)

    if len(X) == 0 or len(X.columns) == 0:
        raise ValueError("Dataset has no featurized candidates. Reload or replace the dataset.")

    # This is a read endpoint: do not initialize or mutate experiments as a
    # side effect of requesting a visualization. Read persisted state only.
    state = _STATE_CACHE.get(experiment_id) or _load_state(experiment_id)
    labeled_ids = (
        {state.pool_ids[i] for i in state.labeled_indices if i < len(state.pool_ids)}
        if state is not None else set()
    )
    initial_observations = (
        {state.pool_ids[i]: float(state.pool_y.iloc[i])
         for i in state.labeled_indices if i < len(state.pool_ids)}
        if state is not None else {}
    )

    # Compute PCA embedding
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    pca = PCA(n_components=n_components, random_state=exp.random_seed)
    coords = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.tolist()

    # Pull latest predictions from DB for overlay
    latest_iter = exp.current_iteration
    preds_q = (
        db.query(CandidatePredictionRecord)
        .filter(
            CandidatePredictionRecord.experiment_id == experiment_id,
            CandidatePredictionRecord.iteration_num == latest_iter,
        )
        .all()
    )
    pred_map = {p.candidate_id: p for p in preds_q}

    # Pull acquired true values
    acq_results = (
        db.query(ExperimentalResult)
        .filter(ExperimentalResult.experiment_id == experiment_id)
        .all()
    )
    acq_map = {r.candidate_id: r.true_value for r in acq_results}

    # Combine into output — limit to 2000 points for frontend performance
    step = max(1, len(X) // 2000)
    points = []
    for i in range(0, len(X), step):
        cid = candidate_ids[i]
        pred = pred_map.get(cid)
        is_obs = cid in labeled_ids
        pt: dict = {
            "candidate_id": cid,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "is_observed": is_obs,
            "true_value": acq_map.get(cid, initial_observations.get(cid)),
            "predicted_mean": pred.predicted_mean if pred else None,
            "predicted_std": pred.predicted_std if pred else None,
            "acquisition_score": pred.acquisition_score if pred else None,
            "is_selected": pred.is_selected if pred else False,
        }
        if n_components >= 3:
            pt["z"] = float(coords[i, 2])
        points.append(pt)

    return {
        "experiment_id": experiment_id,
        "iteration": latest_iter,
        "n_components": n_components,
        "explained_variance": explained,
        "total_explained": sum(explained),
        "method": "PCA",
        "n_pool": len(X),
        "n_shown": len(points),
        "n_observed": len(labeled_ids),
        "points": points,
        "axis_labels": [f"PC{i+1} ({v*100:.1f}%)" for i, v in enumerate(explained)],
    }


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    db: Session,
    experiment_id: str,
    fmt: str = "json",
) -> Tuple[bytes, str, str]:
    """
    Generate a downloadable experiment report.

    Returns (content_bytes, media_type, filename).
    """
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")

    history = get_iteration_history(db, experiment_id)
    discovery = get_discovery_list(db, experiment_id, top_n=100)
    status = get_experiment_status(db, experiment_id)
    dataset = get_dataset_by_id(db, exp.dataset_id)

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "iteration", "budget_used", "budget_remaining",
            "test_mae", "test_rmse", "test_r2",
            "best_found_value", "best_found_id", "regret",
            "diversity_score", "n_selected", "iteration_time_s",
        ])
        for it in history:
            writer.writerow([
                it.iteration_num, it.budget_used, it.budget_remaining,
                it.test_mae, it.test_rmse, it.test_r2,
                it.best_found_value, it.best_found_id, it.regret,
                it.diversity_score, len(it.selected_candidate_ids or []),
                it.iteration_time_seconds,
            ])
        content = output.getvalue().encode()
        return content, "text/csv", f"experiment_{experiment_id[:8]}_history.csv"

    else:  # json
        report_data = {
            "metadata": {
                "experiment_id": experiment_id,
                "name": exp.name,
                "created_at": exp.created_at.isoformat(),
                "generated_at": datetime.utcnow().isoformat(),
                "title": "Uncertainty-Guided Bayesian Active Learning for Data-Efficient Materials Discovery",
            },
            "dataset": {
                "name": dataset.name if dataset else "unknown",
                "source": dataset.source if dataset else "unknown",
                "num_samples": dataset.num_samples if dataset else None,
                "num_features": dataset.num_features if dataset else None,
                "target_column": exp.target_column,
                "license": "CC BY 4.0",
                "note": "Retrospective Simulated Evaluation — true values are from the held-out dataset, not real laboratory experiments.",
            },
            "configuration": {
                "strategy": exp.strategy,
                "model_type": exp.model_type,
                "initial_samples": exp.initial_samples,
                "batch_size": exp.batch_size,
                "total_budget": exp.total_budget,
                "random_seed": exp.random_seed,
                "alpha": exp.alpha, "beta": exp.beta, "gamma": exp.gamma,
                "xi": exp.xi, "kappa": exp.kappa,
                "use_pca": exp.use_pca,
            },
            "results": status,
            "iteration_history": [
                {
                    "iteration": it.iteration_num,
                    "budget_used": it.budget_used,
                    "test_mae": it.test_mae, "test_rmse": it.test_rmse, "test_r2": it.test_r2,
                    "best_found_value": it.best_found_value,
                    "best_found_id": it.best_found_id,
                    "regret": it.regret, "diversity_score": it.diversity_score,
                    "selected_candidates": it.selected_candidate_ids,
                    "selected_true_values": it.selected_true_values,
                    "iteration_time_seconds": it.iteration_time_seconds,
                }
                for it in history
            ],
            "top_discovered_materials": discovery,
            "evaluation_note": (
                "All metrics computed on a fixed 20% test set held out before the AL loop began. "
                "Revealed target values are retrospective dataset lookups, not physical experiments."
            ),
        }
        content = json.dumps(report_data, indent=2, default=str).encode()
        return content, "application/json", f"experiment_{experiment_id[:8]}_report.json"


# ── Private helpers ───────────────────────────────────────────────────────────

def activate_tuned_configuration(
    db: Session, experiment_id: str, tuning_run: TuningRun
) -> None:
    """Apply a completed configuration before an experiment's first iteration."""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if exp is None:
        raise ValueError("Experiment not found.")
    if exp.status != "initialized" or exp.current_iteration != 0:
        raise ValueError("A tuned configuration can only be applied before the first iteration.")
    if tuning_run.experiment_id != experiment_id:
        raise ValueError("Tuning run does not belong to this experiment.")

    state = _STATE_CACHE.get(experiment_id) or _load_state(experiment_id)
    if state is None:
        raise RuntimeError("Initialized experiment state is unavailable.")

    db.query(TuningRun).filter(
        TuningRun.experiment_id == experiment_id,
        TuningRun.applied_to_experiment.is_(True),
    ).update({"applied_to_experiment": False}, synchronize_session=False)
    tuning_run.applied_to_experiment = True
    db.commit()

    config = _exp_to_config(db, exp)
    engine = ActiveLearningEngine(config)
    state.config = config
    _ENGINE_CACHE[experiment_id] = engine
    _STATE_CACHE[experiment_id] = state
    _save_state(experiment_id, state)


def _exp_to_config(db: Session, exp: Experiment) -> ExperimentConfig:
    applied_tuning = (
        db.query(TuningRun)
        .filter(
            TuningRun.experiment_id == exp.id,
            TuningRun.applied_to_experiment.is_(True),
        )
        .order_by(TuningRun.finished_at.desc())
        .first()
    )
    return ExperimentConfig(
        experiment_id=exp.id,
        strategy=exp.strategy, model_type=exp.model_type,
        initial_samples=exp.initial_samples, batch_size=exp.batch_size,
        total_budget=exp.total_budget, random_seed=exp.random_seed,
        xi=exp.xi, kappa=exp.kappa,
        alpha=exp.alpha, beta=exp.beta, gamma=exp.gamma,
        use_pca=exp.use_pca, pca_n_components=exp.pca_n_components,
        model_params=dict(applied_tuning.best_params or {}) if applied_tuning else {},
        target_col=exp.target_column,
    )


def _get_or_restore_engine_state(
    db: Session, exp: Experiment,
) -> Tuple[ActiveLearningEngine, ExperimentState]:
    """
    Return engine+state from in-memory cache, or restore from disk pickle,
    or re-initialize from scratch as last resort.
    """
    eid = exp.id

    # 1. Fast path: in-memory cache
    if eid in _ENGINE_CACHE and eid in _STATE_CACHE:
        engine, state = _ENGINE_CACHE[eid], _STATE_CACHE[eid]
        _upgrade_reference_maxima(db, exp, state)
        return engine, state

    # 2. Restore from disk (survives server restarts)
    state = _load_state(eid)
    if state is not None:
        logger.info("Restored state from disk for experiment %s", eid)
        _upgrade_reference_maxima(db, exp, state)
        config = _exp_to_config(db, exp)
        engine = ActiveLearningEngine(config)
        _ENGINE_CACHE[eid] = engine
        _STATE_CACHE[eid] = state
        return engine, state

    # 3. Full re-initialization (new random seed continuity is lost but better than crashing)
    logger.warning("No saved state for %s — re-initializing from scratch.", eid)
    initialize_experiment(db, eid)
    return _ENGINE_CACHE[eid], _STATE_CACHE[eid]


def _dataset_maximum(db: Session, exp: Experiment) -> float:
    """Get the retrospective full-dataset maximum, repairing legacy metadata if needed."""
    dataset = get_dataset_by_id(db, exp.dataset_id)
    if dataset and dataset.global_maximum is not None:
        return float(dataset.global_maximum)
    if dataset:
        _, y, _ = load_dataset_arrays(dataset)
        return float(y.max())
    return float(exp.global_maximum or 0.0)


def _upgrade_reference_maxima(db: Session, exp: Experiment, state: ExperimentState) -> None:
    """Upgrade pre-existing pickled states without reinitializing their experiments."""
    if not hasattr(state, "training_pool_maximum"):
        state.training_pool_maximum = float(state.pool_y.max())
    reference_maximum = _dataset_maximum(db, exp)
    state.global_maximum = reference_maximum
    exp.global_maximum = reference_maximum
    for iteration in db.query(ExperimentIteration).filter(
        ExperimentIteration.experiment_id == exp.id
    ):
        if iteration.best_found_value is not None:
            iteration.regret = reference_maximum - iteration.best_found_value
    db.commit()
    _save_state(exp.id, state)
