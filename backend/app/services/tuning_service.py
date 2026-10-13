"""Optuna-based model hyperparameter tuning on initially labeled data only."""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from backend.app.models.dataset import Dataset
from backend.app.models.experiment import Experiment, TuningRun
from backend.app.services.dataset_service import get_dataset_by_id, load_dataset_arrays
from backend.database import SessionLocal
from backend.ml.data.preprocessing import PreprocessingConfig, fit_preprocessor
from backend.ml.models.baseline import create_surrogate
from backend.ml.models.bayesian import create_bayesian_surrogate

logger = logging.getLogger(__name__)
SEARCH_SPACE_VERSION = "rf-gp-v1"
VALIDATION_FRACTION = 0.20


def create_tuning_run(
    db: Session,
    experiment_id: str,
    model_type: str,
    n_trials: int,
    random_seed: int,
    timeout_seconds: int | None,
) -> TuningRun:
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if experiment is None:
        raise ValueError("Experiment not found.")
    if experiment.status != "initialized" or experiment.current_iteration != 0:
        raise ValueError("Tuning is only available before the first active-learning iteration.")
    if model_type != experiment.model_type:
        raise ValueError("Tuning model_type must match the experiment model_type.")
    if model_type not in ("rf", "gp"):
        raise ValueError("Tuning currently supports Random Forest (rf) and Gaussian Process (gp).")
    active_run = (
        db.query(TuningRun.id)
        .filter(
            TuningRun.experiment_id == experiment_id,
            TuningRun.status.in_(("pending", "running")),
        )
        .first()
    )
    if active_run is not None:
        raise ValueError("A tuning run is already active for this experiment.")

    run = TuningRun(
        experiment_id=experiment_id,
        model_type=model_type,
        trial_count=n_trials,
        random_seed=random_seed,
        timeout_seconds=timeout_seconds,
        search_space_version=SEARCH_SPACE_VERSION,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_tuning_run(db: Session, experiment_id: str, run_id: str) -> TuningRun:
    run = (
        db.query(TuningRun)
        .filter(TuningRun.id == run_id, TuningRun.experiment_id == experiment_id)
        .first()
    )
    if run is None:
        raise ValueError("Tuning run not found.")
    return run


def run_tuning_background(run_id: str) -> None:
    """Run tuning in FastAPI's background task with an independent DB session."""
    db = SessionLocal()
    run = db.query(TuningRun).filter(TuningRun.id == run_id).first()
    if run is None:
        db.close()
        logger.error("Cannot start missing tuning run %s", run_id)
        return

    try:
        experiment = db.query(Experiment).filter(Experiment.id == run.experiment_id).first()
        if experiment is None:
            raise ValueError("Associated experiment no longer exists.")
        if experiment.status != "initialized" or experiment.current_iteration != 0:
            raise ValueError("Experiment must remain initialized and have no completed iterations.")

        dataset = get_dataset_by_id(db, experiment.dataset_id)
        if dataset is None or dataset.status != "ready":
            raise ValueError("Experiment dataset is not available.")
        X, y, _ = load_dataset_arrays(dataset)
        pool_indices, labeled_indices = _initial_labeled_indices(
            len(X), experiment.initial_samples, experiment.random_seed
        )
        labeled_X = X.iloc[pool_indices[labeled_indices]].reset_index(drop=True)
        labeled_y = y.iloc[pool_indices[labeled_indices]].to_numpy(dtype=float)
        if len(labeled_X) < 5:
            raise ValueError("At least five initially labeled samples are required for tuning.")

        indices = np.arange(len(labeled_X))
        train_idx, validation_idx = train_test_split(
            indices,
            test_size=max(2, math.ceil(len(indices) * VALIDATION_FRACTION)),
            random_state=run.random_seed,
            shuffle=True,
        )
        X_train = labeled_X.iloc[train_idx]
        y_train = labeled_y[train_idx]
        X_validation = labeled_X.iloc[validation_idx]
        y_validation = labeled_y[validation_idx]

        run.status = "running"
        run.started_at = datetime.utcnow()
        run.training_size = int(len(train_idx))
        run.validation_size = int(len(validation_idx))
        run.dataset_fingerprint = _feature_fingerprint(dataset, X)
        run.split_metadata = {
            "strategy": "random_train_validation_split",
            "validation_fraction": VALIDATION_FRACTION,
            "random_seed": run.random_seed,
            "initial_pool_seed": experiment.random_seed,
            "source": "initially_labeled_training_pool_only",
            "initial_samples": int(len(labeled_indices)),
            "training_size": int(len(train_idx)),
            "validation_size": int(len(validation_idx)),
        }
        db.commit()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=run.random_seed),
            study_name=f"materials-tuning-{run.id}",
        )

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial, run.model_type)
            preprocessing = fit_preprocessor(
                X_train,
                PreprocessingConfig(
                    random_state=run.random_seed,
                    use_pca=experiment.use_pca,
                    pca_n_components=experiment.pca_n_components,
                ),
                labeled_set_size=len(X_train),
            )
            transformed_train = preprocessing.transform(X_train)
            transformed_validation = preprocessing.transform(X_validation)
            model = _build_model(run.model_type, params, run.random_seed)
            model.fit(transformed_train, y_train)
            mean, _ = model.predict(transformed_validation)
            rmse = float(np.sqrt(mean_squared_error(y_validation, mean)))
            trial.set_user_attr(
                "validation_metrics",
                {
                    "rmse": rmse,
                    "mae": float(mean_absolute_error(y_validation, mean)),
                    "r2": float(r2_score(y_validation, mean)),
                },
            )
            return rmse

        def save_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            run.completed_trials = sum(
                item.state == optuna.trial.TrialState.COMPLETE for item in study.trials
            )
            run.failed_trials = sum(
                item.state == optuna.trial.TrialState.FAIL for item in study.trials
            )
            run.pruned_trials = sum(
                item.state == optuna.trial.TrialState.PRUNED for item in study.trials
            )
            run.trial_history = _trial_history(study)
            if study.best_trials:
                run.best_params = dict(study.best_trial.params)
                run.best_validation_rmse = float(study.best_value)
                metrics = study.best_trial.user_attrs.get("validation_metrics", {})
                run.best_validation_mae = metrics.get("mae")
                run.best_validation_r2 = metrics.get("r2")
            db.commit()

        study.optimize(
            objective,
            n_trials=run.trial_count,
            timeout=run.timeout_seconds,
            callbacks=[save_progress],
            catch=(Exception,),
            show_progress_bar=False,
        )
        run.trial_history = _trial_history(study)
        run.completed_trials = sum(
            item.state == optuna.trial.TrialState.COMPLETE for item in study.trials
        )
        run.failed_trials = sum(
            item.state == optuna.trial.TrialState.FAIL for item in study.trials
        )
        run.pruned_trials = sum(
            item.state == optuna.trial.TrialState.PRUNED for item in study.trials
        )
        if run.completed_trials == 0:
            run.status = "failed"
            run.error_summary = "No tuning trial completed successfully."
        else:
            best = study.best_trial
            run.status = "completed"
            run.best_params = dict(best.params)
            run.best_validation_rmse = float(best.value)
            metrics = best.user_attrs.get("validation_metrics", {})
            run.best_validation_mae = metrics.get("mae")
            run.best_validation_r2 = metrics.get("r2")
            if run.failed_trials:
                run.error_summary = f"{run.failed_trials} trial(s) failed; successful trials were retained."
        run.finished_at = datetime.utcnow()
        db.commit()
    except Exception as exc:
        db.rollback()
        run = db.query(TuningRun).filter(TuningRun.id == run_id).first()
        if run is not None:
            run.status = "failed"
            run.error_summary = "Tuning run failed. See backend logs for diagnostic details."
            run.finished_at = datetime.utcnow()
            db.commit()
        logger.exception("Tuning run %s failed", run_id)
    finally:
        db.close()


def _initial_labeled_indices(
    n_samples: int, initial_samples: int, random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce engine initialization without reading any unselected labels."""
    indices = np.arange(n_samples)
    pool_indices, _ = train_test_split(
        indices, test_size=0.20, random_state=random_seed, shuffle=True
    )
    n_initial = min(initial_samples, len(pool_indices) - 1)
    rng = np.random.RandomState(random_seed)
    labeled_indices = rng.choice(len(pool_indices), size=n_initial, replace=False)
    return pool_indices, labeled_indices


def _suggest_params(trial: optuna.Trial, model_type: str) -> dict[str, Any]:
    if model_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 16, 32]),
            "min_samples_split": trial.suggest_categorical(
                "min_samples_split", [2, 4, 8]
            ),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf", [1, 2, 4]
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.5, 1.0]
            ),
        }
    if model_type == "gp":
        return {
            "length_scale": trial.suggest_float("length_scale", 0.1, 10.0, log=True),
            "noise_level": trial.suggest_float("noise_level", 1e-5, 1.0, log=True),
            "n_restarts_optimizer": trial.suggest_int(
                "n_restarts_optimizer", 0, 1
            ),
            "normalize_y": trial.suggest_categorical("normalize_y", [True, False]),
        }
    raise ValueError("Tuning currently supports only rf and gp.")


def _build_model(model_type: str, params: dict[str, Any], random_seed: int):
    if model_type == "rf":
        return create_surrogate(model_type, random_state=random_seed, **params)
    return create_bayesian_surrogate(model_type, random_state=random_seed, **params)


def _trial_history(study: optuna.Study) -> list[dict[str, Any]]:
    history = []
    for trial in study.trials[-100:]:
        item: dict[str, Any] = {
            "number": trial.number + 1,
            "state": trial.state.name.lower(),
            "value": float(trial.value) if trial.value is not None else None,
            "params": trial.params,
        }
        if trial.state == optuna.trial.TrialState.FAIL:
            item["error"] = str(trial.system_attrs.get("fail_reason", "Trial failed."))[:300]
        history.append(item)
    return history


def _feature_fingerprint(dataset: Dataset, X) -> str:
    """Fingerprint dataset identity and features without reading any target labels."""
    digest = hashlib.sha256()
    digest.update(str(dataset.id).encode())
    digest.update("\0".join(map(str, X.columns)).encode())
    digest.update(np.ascontiguousarray(X.to_numpy()).tobytes())
    return digest.hexdigest()


def apply_tuning_run(db: Session, experiment_id: str, run_id: str) -> TuningRun:
    from backend.app.services.experiment_service import (
        activate_tuned_configuration,
    )

    run = get_tuning_run(db, experiment_id, run_id)
    if run.status != "completed" or not run.best_params:
        raise ValueError("Only successfully completed tuning runs can be applied.")
    activate_tuned_configuration(db, experiment_id, run)
    return run
