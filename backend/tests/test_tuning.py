"""Tests for Optuna model tuning and tuned active-learning configurations."""

import pickle
import uuid

import numpy as np
import optuna
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.models.dataset import Dataset
from backend.app.models.experiment import Experiment, TuningRun
from backend.app.services import experiment_service, tuning_service
from backend.database import Base
from backend.ml.active_learning.engine import ActiveLearningEngine, ExperimentConfig


@pytest.fixture
def tuning_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'tuning.sqlite'}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(tuning_service, "SessionLocal", session_factory)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


def _add_experiment(session, tmp_path, model_type="rf", initial_samples=20):
    rng = np.random.RandomState(5)
    X = pd.DataFrame(rng.normal(size=(100, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(1.4 * X["f0"] - 0.8 * X["f1"] + rng.normal(0, 0.1, len(X)))
    ids = [f"MAT-{i:04d}" for i in range(len(X))]
    cache_path = tmp_path / f"{uuid.uuid4()}.pkl"
    with open(cache_path, "wb") as cache_file:
        pickle.dump({"X": X, "y": y, "candidate_ids": ids}, cache_file)

    dataset_id = str(uuid.uuid4())
    session.add(Dataset(
        id=dataset_id,
        name="synthetic",
        status="ready",
        cache_path=str(cache_path),
        num_samples=len(X),
        num_features=X.shape[1],
        feature_columns=list(X.columns),
        global_maximum=float(y.max()),
        global_minimum=float(y.min()),
    ))
    experiment_id = str(uuid.uuid4())
    session.add(Experiment(
        id=experiment_id,
        dataset_id=dataset_id,
        model_type=model_type,
        strategy="random",
        initial_samples=initial_samples,
        batch_size=5,
        total_budget=40,
        random_seed=42,
        current_iteration=0,
        status="initialized",
    ))
    session.commit()
    return experiment_id, X, y


def _run_tuning(session, experiment_id, model_type, n_trials=2):
    run = tuning_service.create_tuning_run(
        session,
        experiment_id=experiment_id,
        model_type=model_type,
        n_trials=n_trials,
        random_seed=123,
        timeout_seconds=60,
    )
    tuning_service.run_tuning_background(run.id)
    session.expire_all()
    return session.query(TuningRun).filter(TuningRun.id == run.id).one()


@pytest.mark.parametrize("model_type", ["rf", "gp"])
def test_real_tuning_workflow_minimizes_validation_rmse(
    tuning_session, tmp_path, model_type
):
    experiment_id, _, _ = _add_experiment(
        tuning_session, tmp_path, model_type=model_type
    )
    run = _run_tuning(tuning_session, experiment_id, model_type)

    assert run.status == "completed"
    assert run.best_params
    assert run.completed_trials == 2
    assert run.failed_trials == 0
    assert run.best_validation_rmse == min(
        item["value"] for item in run.trial_history if item["value"] is not None
    )
    assert run.training_size + run.validation_size == 20
    assert run.split_metadata["source"] == "initially_labeled_training_pool_only"
    assert run.dataset_fingerprint


def test_changing_unlabeled_and_test_targets_does_not_change_tuning(
    tuning_session, tmp_path
):
    experiment_id, _, _ = _add_experiment(tuning_session, tmp_path)
    first = _run_tuning(tuning_session, experiment_id, "rf", n_trials=2)

    experiment = tuning_session.query(Experiment).filter(
        Experiment.id == experiment_id
    ).one()
    dataset = tuning_session.query(Dataset).filter(
        Dataset.id == experiment.dataset_id
    ).one()
    with open(dataset.cache_path, "rb") as cache_file:
        cache = pickle.load(cache_file)
    y = cache["y"].copy()
    pool_indices, labeled_indices = tuning_service._initial_labeled_indices(
        len(y), experiment.initial_samples, experiment.random_seed
    )
    selected_rows = set(pool_indices[labeled_indices])
    unknown_rows = [i for i in range(len(y)) if i not in selected_rows]
    y.iloc[unknown_rows] = 10_000.0
    cache["y"] = y
    with open(dataset.cache_path, "wb") as cache_file:
        pickle.dump(cache, cache_file)

    second = _run_tuning(tuning_session, experiment_id, "rf", n_trials=2)
    assert first.best_validation_rmse == second.best_validation_rmse
    assert first.best_params == second.best_params


def test_optuna_search_is_reproducible_for_fixed_seed():
    def sampled_params(seed):
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=seed),
            direction="minimize",
        )
        values = []
        for _ in range(4):
            trial = study.ask()
            values.append(tuning_service._suggest_params(trial, "rf"))
            study.tell(trial, float(trial.number))
        return values

    assert sampled_params(71) == sampled_params(71)
    assert sampled_params(71) != sampled_params(72)


def test_initial_seed_selection_does_not_include_test_indices():
    pool_indices, labeled_indices = tuning_service._initial_labeled_indices(
        n_samples=100, initial_samples=20, random_seed=42
    )
    repeat_pool, repeat_labeled = tuning_service._initial_labeled_indices(
        n_samples=100, initial_samples=20, random_seed=42
    )
    assert np.array_equal(pool_indices, repeat_pool)
    assert np.array_equal(labeled_indices, repeat_labeled)
    assert len(labeled_indices) == 20
    assert set(pool_indices[labeled_indices]).issubset(set(pool_indices))
    test_indices = set(range(100)) - set(pool_indices)
    assert not test_indices.intersection(pool_indices[labeled_indices])


def test_failed_trials_are_persisted_without_aborting_worker(
    tuning_session, tmp_path, monkeypatch
):
    experiment_id, _, _ = _add_experiment(tuning_session, tmp_path)
    monkeypatch.setattr(
        tuning_service,
        "_build_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    run = _run_tuning(tuning_session, experiment_id, "rf", n_trials=1)
    assert run.status == "failed"
    assert run.failed_trials == 1
    assert run.error_summary == "No tuning trial completed successfully."
    assert run.trial_history[0]["state"] == "fail"


def test_pruned_trials_are_counted_and_persisted(
    tuning_session, tmp_path, monkeypatch
):
    experiment_id, _, _ = _add_experiment(tuning_session, tmp_path)
    monkeypatch.setattr(
        tuning_service,
        "_suggest_params",
        lambda *args, **kwargs: (_ for _ in ()).throw(optuna.TrialPruned()),
    )
    run = _run_tuning(tuning_session, experiment_id, "rf", n_trials=1)
    assert run.status == "failed"
    assert run.pruned_trials == 1
    assert run.trial_history[0]["state"] == "pruned"


def test_tuned_params_are_passed_into_rf_training():
    params = {
        "n_estimators": 50,
        "max_depth": 8,
        "min_samples_split": 4,
        "min_samples_leaf": 1,
        "max_features": 0.5,
    }
    engine = ActiveLearningEngine(
        ExperimentConfig(model_type="rf", model_params=params)
    )
    model = engine._create_model()
    assert model.n_estimators == 50
    assert model.max_depth == 8
    assert model.min_samples_split == 4
    assert model.min_samples_leaf == 1
    assert model.max_features == 0.5


def test_tuned_params_are_passed_into_gp_training():
    params = {
        "length_scale": 2.5,
        "noise_level": 0.02,
        "n_restarts_optimizer": 1,
        "normalize_y": False,
    }
    engine = ActiveLearningEngine(
        ExperimentConfig(model_type="gp", model_params=params)
    )
    model = engine._create_model()
    assert model.length_scale == 2.5
    assert model.noise_level == 0.02
    assert model.n_restarts_optimizer == 1
    assert model.normalize_y is False


def test_applying_tuning_persists_params_and_preserves_manual_defaults(
    tuning_session, tmp_path, monkeypatch
):
    experiment_id, _, _ = _add_experiment(tuning_session, tmp_path)
    run = _run_tuning(tuning_session, experiment_id, "rf", n_trials=1)
    monkeypatch.setattr(experiment_service, "_save_state", lambda *args: None)
    state = experiment_service.ExperimentState(
        config=ExperimentConfig(experiment_id=experiment_id, model_type="rf")
    )
    monkeypatch.setitem(experiment_service._STATE_CACHE, experiment_id, state)
    monkeypatch.setitem(experiment_service._ENGINE_CACHE, experiment_id, None)

    tuning_service.apply_tuning_run(tuning_session, experiment_id, run.id)
    tuned_engine = experiment_service._ENGINE_CACHE[experiment_id]
    assert tuned_engine.config.model_params == run.best_params

    second_experiment_id, _, _ = _add_experiment(tuning_session, tmp_path)
    second = tuning_session.query(Experiment).filter(
        Experiment.id == second_experiment_id
    ).one()
    manual_config = experiment_service._exp_to_config(tuning_session, second)
    assert manual_config.model_params == {}
