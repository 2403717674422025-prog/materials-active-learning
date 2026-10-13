"""Experiment routes — all endpoints for the AL experiment lifecycle."""

from __future__ import annotations

import math
import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.schemas.candidate import CandidateDetail, CandidateListResponse, CandidatePrediction
from backend.app.schemas.experiment import (
    ExperimentCreate, ExperimentResponse, ExperimentStatus,
    IterationHistoryItem, IterationRequest, IterationResponse,
)
from backend.app.schemas.tuning import TuningRunResponse, TuningStartRequest
from backend.app.services.experiment_service import (
    create_experiment, get_candidate_detail, get_candidates_for_iteration,
    get_discovery_list, get_experiment_status, get_iteration_history,
    initialize_experiment, list_experiments, run_iteration,
    get_experiment_embedding, generate_report,
)
from backend.app.services.tuning_service import (
    apply_tuning_run, create_tuning_run, get_tuning_run, run_tuning_background,
)
from backend.app.models.experiment import Experiment, TuningRun
from backend.database import get_db
from backend.app.services import explainability_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _explainability_request(call):
    try:
        return call()
    except explainability_service.ExplainabilityError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Explainability request failed")
        raise HTTPException(status_code=500, detail="Explainability request failed.") from exc


# ── List / create ─────────────────────────────────────────────────────────────

@router.get("/", response_model=List[ExperimentResponse])
def list_all(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List all experiments, most-recent first."""
    return [ExperimentResponse.model_validate(e) for e in list_experiments(db, skip, limit)]


@router.post("/", response_model=ExperimentResponse, status_code=201)
def create(req: ExperimentCreate, db: Session = Depends(get_db)):
    """Create a new active learning experiment configuration."""
    try:
        return ExperimentResponse.model_validate(create_experiment(db, req))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@router.post("/{experiment_id}/initialize")
def initialize(experiment_id: str, db: Session = Depends(get_db)):
    """Initialize: seed labeled set, persist state to disk."""
    try:
        return initialize_experiment(db, experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {exc}")


@router.post("/{experiment_id}/run-iteration", response_model=IterationResponse)
def run_one_iteration(
    experiment_id: str,
    req: IterationRequest = IterationRequest(),
    db: Session = Depends(get_db),
):
    """Execute one full AL iteration: train→predict→score→select→acquire→update→evaluate."""
    try:
        r = run_iteration(db, experiment_id)
        return IterationResponse(
            iteration=r.iteration, experiment_id=r.experiment_id,
            strategy=r.strategy, model_type=r.model_type,
            budget_used=r.budget_used, budget_remaining=r.budget_remaining,
            selected_candidate_ids=r.selected_candidate_ids,
            selected_true_values=r.selected_true_values,
            test_mae=r.test_mae, test_rmse=r.test_rmse, test_r2=r.test_r2,
            best_found_value=r.best_found_value, best_found_id=r.best_found_id,
            regret=r.regret, global_maximum=r.global_maximum,
            training_pool_maximum=r.training_pool_maximum,
            mean_uncertainty_selected=r.mean_uncertainty_selected,
            diversity_score=r.diversity_score,
            iteration_time_seconds=r.iteration_time_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Iteration failed: {exc}")


@router.post(
    "/{experiment_id}/tuning-runs",
    response_model=TuningRunResponse,
    status_code=202,
)
def start_tuning(
    experiment_id: str,
    req: TuningStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start bounded RF/GP tuning using only the experiment's initial labeled samples."""
    try:
        run = create_tuning_run(
            db=db,
            experiment_id=experiment_id,
            model_type=req.model_type,
            n_trials=req.n_trials,
            random_seed=req.random_seed,
            timeout_seconds=req.timeout_seconds,
        )
        background_tasks.add_task(run_tuning_background, run.id)
        return TuningRunResponse.model_validate(run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/{experiment_id}/tuning-runs",
    response_model=List[TuningRunResponse],
)
def list_tuning_runs(experiment_id: str, db: Session = Depends(get_db)):
    experiment = db.query(Experiment.id).filter(Experiment.id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    runs = (
        db.query(TuningRun)
        .filter(TuningRun.experiment_id == experiment_id)
        .order_by(TuningRun.created_at.desc())
        .all()
    )
    return [TuningRunResponse.model_validate(run) for run in runs]


@router.get(
    "/{experiment_id}/tuning-runs/{run_id}",
    response_model=TuningRunResponse,
)
def tuning_status(experiment_id: str, run_id: str, db: Session = Depends(get_db)):
    try:
        return TuningRunResponse.model_validate(get_tuning_run(db, experiment_id, run_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{experiment_id}/tuning-runs/{run_id}/apply",
    response_model=TuningRunResponse,
)
def apply_tuning(experiment_id: str, run_id: str, db: Session = Depends(get_db)):
    """Apply the best validation-selected model parameters before AL starts."""
    try:
        run = apply_tuning_run(db, experiment_id, run_id)
        return TuningRunResponse.model_validate(run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Could not apply tuning run %s", run_id)
        raise HTTPException(status_code=500, detail="Could not apply tuned configuration.") from exc


# ── Query / inspection ────────────────────────────────────────────────────────

@router.get("/{experiment_id}/status", response_model=ExperimentStatus)
def experiment_status(experiment_id: str, db: Session = Depends(get_db)):
    try:
        return ExperimentStatus(**get_experiment_status(db, experiment_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{experiment_id}/history", response_model=List[IterationHistoryItem])
def iteration_history(experiment_id: str, db: Session = Depends(get_db)):
    iters = get_iteration_history(db, experiment_id)
    return [
        IterationHistoryItem(
            iteration=it.iteration_num,
            budget_used=it.budget_used or 0,
            budget_remaining=it.budget_remaining or 0,
            test_mae=it.test_mae, test_rmse=it.test_rmse, test_r2=it.test_r2,
            best_found_value=it.best_found_value, best_found_id=it.best_found_id,
            regret=it.regret, diversity_score=it.diversity_score,
            n_candidates_selected=len(it.selected_candidate_ids or []),
            iteration_time_seconds=it.iteration_time_seconds,
            timestamp=it.timestamp,
        )
        for it in iters
    ]


@router.get("/{experiment_id}/candidates", response_model=CandidateListResponse)
def list_candidates(
    experiment_id: str,
    iteration: Optional[int] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    records, total = get_candidates_for_iteration(db, experiment_id, iteration, page, page_size)
    from backend.app.models.experiment import Experiment
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    current_iter = exp.current_iteration if exp else 0
    calibrated_intervals = bool(exp and exp.model_type in ("gp", "gaussian_process"))

    candidates = []
    for r in records:
        std = r.predicted_std or 0.0
        ul = "Low" if std < 0.3 else ("Medium" if std < 0.8 else "High")
        candidates.append(CandidatePrediction(
            candidate_id=r.candidate_id, pool_index=r.pool_index or 0,
            predicted_mean=r.predicted_mean or 0.0, predicted_std=std,
            lower_cb=r.lower_cb if calibrated_intervals else None,
            upper_cb=r.upper_cb if calibrated_intervals else None,
            acquisition_score=r.acquisition_score or 0.0,
            is_selected=r.is_selected, selection_rank=r.selection_rank,
            uncertainty_label=ul, breakdown=r.breakdown or {},
        ))
    return CandidateListResponse(
        experiment_id=experiment_id, iteration=iteration or current_iter,
        total_candidates=total, selected_count=sum(1 for c in candidates if c.is_selected),
        candidates=candidates, page=page, page_size=page_size,
        total_pages=math.ceil(total / page_size) if total else 1,
    )


@router.get("/{experiment_id}/candidates/{candidate_id}", response_model=CandidateDetail)
def candidate_detail(experiment_id: str, candidate_id: str, db: Session = Depends(get_db)):
    record = get_candidate_detail(db, experiment_id, candidate_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found.")
    bd = record.breakdown or {}
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    calibrated_intervals = bool(exp and exp.model_type in ("gp", "gaussian_process"))
    return CandidateDetail(
        candidate_id=record.candidate_id, iteration=record.iteration_num,
        predicted_mean=record.predicted_mean or 0.0,
        predicted_std=record.predicted_std or 0.0,
        lower_cb=record.lower_cb if calibrated_intervals else None,
        upper_cb=record.upper_cb if calibrated_intervals else None,
        strategy=bd.get("strategy", "unknown"),
        acquisition_score=record.acquisition_score or 0.0,
        predicted_performance=bd.get("predicted_performance"),
        uncertainty=bd.get("uncertainty"),
        exploration_contribution=bd.get("exploration_contribution"),
        exploitation_contribution=bd.get("exploitation_contribution"),
        diversity_contribution=bd.get("diversity_contribution"),
        formula=bd.get("formula"), selection_reason=bd.get("selection_reason", ""),
        is_selected=record.is_selected, selection_rank=record.selection_rank,
        full_breakdown=bd,
    )


@router.get("/{experiment_id}/discovery")
def discovery_list(
    experiment_id: str,
    top_n: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return get_discovery_list(db, experiment_id, top_n=top_n)


@router.get("/{experiment_id}/explainability/candidates")
def explainability_candidates(
    experiment_id: str,
    q: str = Query(default="", max_length=256),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Search candidate IDs in the AL pool without reading their target values."""
    return _explainability_request(
        lambda: explainability_service.list_explainability_candidates(
            db, experiment_id, q, limit
        )
    )


@router.get("/{experiment_id}/explainability/global")
def explainability_global(
    experiment_id: str,
    sample_size: int = Query(default=128, ge=1, le=256),
    db: Session = Depends(get_db),
):
    """Return global RF SHAP importance and beeswarm points from the AL pool."""
    return _explainability_request(
        lambda: explainability_service.get_global_explanation(
            db, experiment_id, sample_size
        )
    )


@router.get("/{experiment_id}/explainability/candidates/{candidate_id}")
def explainability_candidate(
    experiment_id: str,
    candidate_id: str,
    db: Session = Depends(get_db),
):
    """Explain one active-learning-pool candidate with the stored fitted model."""
    return _explainability_request(
        lambda: explainability_service.get_candidate_explanation(
            db, experiment_id, candidate_id
        )
    )


@router.get("/{experiment_id}/explainability/dependence")
def explainability_dependence(
    experiment_id: str,
    feature: str = Query(min_length=1, max_length=256),
    sample_size: int = Query(default=128, ge=1, le=256),
    db: Session = Depends(get_db),
):
    """Return SHAP dependence points for a real model input feature."""
    return _explainability_request(
        lambda: explainability_service.get_feature_dependence(
            db, experiment_id, feature, sample_size
        )
    )


# ── Visualization: feature-space embedding ────────────────────────────────────

@router.get("/{experiment_id}/embedding")
def embedding(
    experiment_id: str,
    n_components: int = Query(default=2, ge=2, le=3),
    db: Session = Depends(get_db),
):
    """
    Return PCA-projected 2D/3D coordinates for all pool candidates.
    Includes predicted mean, std, acquisition score, and selection status
    so the frontend can color-code the scatter plot.

    PCA is fitted on ALL pool features (not just labeled) to give a
    stable coordinate system. Labels/predictions are overlaid on top.
    """
    try:
        return get_experiment_embedding(db, experiment_id, n_components=n_components)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")


# ── Reports ───────────────────────────────────────────────────────────────────

@router.get("/{experiment_id}/report")
def report(
    experiment_id: str,
    fmt: str = Query(default="json", pattern="^(json|csv)$"),
    db: Session = Depends(get_db),
):
    """
    Download a complete experiment report.
    fmt=json  — full JSON report (config, iterations, metrics, top candidates)
    fmt=csv   — CSV of per-iteration metrics
    """
    try:
        content, media_type, filename = generate_report(db, experiment_id, fmt=fmt)
        return StreamingResponse(
            iter([content]),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{experiment_id}/export")
def export_results(experiment_id: str, db: Session = Depends(get_db)):
    """Legacy export endpoint — returns full JSON."""
    from fastapi.responses import JSONResponse
    history = get_iteration_history(db, experiment_id)
    discovery = get_discovery_list(db, experiment_id, top_n=500)
    status_data = get_experiment_status(db, experiment_id)
    return JSONResponse({
        "experiment": status_data,
        "iterations": [
            {
                "iteration": it.iteration_num, "budget_used": it.budget_used,
                "test_rmse": it.test_rmse, "test_r2": it.test_r2,
                "best_found": it.best_found_value, "regret": it.regret,
                "selected": it.selected_candidate_ids,
            }
            for it in history
        ],
        "top_discovered": discovery,
    })


# ── Benchmark / ablation / calibration ───────────────────────────────────────
# These are compute-heavy. For small datasets / RF model they run synchronously.
# For GP or large datasets the caller should expect a longer response time.

@router.post("/{experiment_id}/benchmark")
def run_benchmark(
    experiment_id: str,
    strategies: Optional[List[str]] = Query(default=None),
    seeds: str = Query(default="", description="Comma-separated integer seeds, e.g. 42,123,456"),
    model_type: str = Query(default="rf", pattern="^(rf|gp|ensemble)$"),
    db: Session = Depends(get_db),
):
    """
    Run a matched multi-strategy benchmark under identical conditions.

    All strategies use the same dataset, initial_samples, batch_size,
    total_budget, and initial labeled set as the parent experiment.

    Use model_type=rf for speed (recommended for benchmarks).
    Use model_type=gp only for small budgets (≤ 200 labeled samples).
    """
    from backend.app.services.benchmark_service import run_strategy_benchmark
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()] if seeds.strip() else None
    try:
        return run_strategy_benchmark(
            db, experiment_id,
            strategies=strategies or None,
            seeds=seed_list,
            model_type=model_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc}")


@router.get("/{experiment_id}/benchmarks/{benchmark_id}")
def get_benchmark(
    experiment_id: str,
    benchmark_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve an immutable persisted benchmark result."""
    from backend.app.services.benchmark_service import get_benchmark_run
    try:
        return get_benchmark_run(db, experiment_id, benchmark_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{experiment_id}/ablation")
def run_ablation(
    experiment_id: str,
    seeds: str = Query(default="", description="Comma-separated seeds"),
    model_type: str = Query(default="rf", pattern="^(rf|gp|ensemble)$"),
    db: Session = Depends(get_db),
):
    """
    Run a PUDA ablation study comparing 4 component configurations:
    pred_only, pred_unc, pred_div, full PUDA.
    """
    from backend.app.services.benchmark_service import run_ablation_study
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()] if seeds.strip() else None
    try:
        return run_ablation_study(
            db, experiment_id,
            seeds=seed_list,
            model_type=model_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ablation failed: {exc}")


@router.get("/{experiment_id}/calibration")
def calibration_metrics(experiment_id: str, db: Session = Depends(get_db)):
    """
    Return uncertainty calibration metrics (ECE, coverage at 90% and 95%)
    for the latest iteration.

    Calibration is computed on the OBSERVED candidates only (not the
    unobserved pool) to avoid revealing held-out targets.
    """
    from backend.app.services.benchmark_service import get_calibration_metrics
    try:
        return get_calibration_metrics(db, experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
