"""
Tests for the Active Learning Engine.

Covers:
  - initialize() sets up correct state
  - run_iteration() produces IterationResult with expected fields
  - budget is respected (never exceeds total_budget)
  - no duplicate candidate selection across iterations
  - all four strategies complete an iteration without error
  - evaluate_iteration() produces correct metric keys
"""

import numpy as np
import pandas as pd
import pytest

from backend.ml.active_learning.engine import (
    ActiveLearningEngine,
    ExperimentConfig,
)


@pytest.fixture
def small_dataset():
    rng = np.random.RandomState(7)
    N = 200
    X = pd.DataFrame(rng.randn(N, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.uniform(0, 10, N), name="target")
    ids = [f"M{i:04d}" for i in range(N)]
    return X, y, ids


def _make_engine(strategy: str, model_type: str = "rf") -> ActiveLearningEngine:
    cfg = ExperimentConfig(
        strategy=strategy,
        model_type=model_type,
        initial_samples=20,
        batch_size=5,
        total_budget=40,
        random_seed=42,
        n_estimators=10,   # fast RF
    )
    return ActiveLearningEngine(cfg)


def _split_dataset(X, y, ids):
    from backend.ml.data.preprocessing import initial_split
    return initial_split(X, y, ids, test_size=0.20, random_state=42)


@pytest.mark.parametrize("strategy", ["random", "uncertainty", "ei", "puda"])
def test_initialize_creates_correct_state(strategy, small_dataset):
    X, y, ids = small_dataset
    split = _split_dataset(X, y, ids)
    engine = _make_engine(strategy)
    state = engine.initialize(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        split["test_X"], split["test_y"], split["test_ids"],
    )
    assert state.is_initialized
    assert len(state.labeled_indices) == 20
    assert len(state.pool_X) == len(split["pool_X"])
    assert state.global_maximum == float(max(split["pool_y"].max(), split["test_y"].max()))
    assert state.training_pool_maximum == float(split["pool_y"].max())


def test_regret_uses_full_dataset_maximum_when_it_is_held_out(small_dataset):
    X, _, ids = small_dataset
    y = pd.Series(np.zeros(len(X)), name="target")
    membership = _split_dataset(X, y, ids)
    held_out_id = membership["test_ids"][0]
    y.iloc[int(held_out_id[1:])] = 100.0
    split = _split_dataset(X, y, ids)
    assert float(split["test_y"].max()) == 100.0
    assert float(split["pool_y"].max()) < 100.0

    engine = _make_engine("random")
    state = engine.initialize(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        split["test_X"], split["test_y"], split["test_ids"],
        dataset_maximum=float(y.max()),
    )
    result, _ = engine.run_iteration(state)

    assert result.global_maximum == 100.0
    assert result.training_pool_maximum == float(split["pool_y"].max())
    assert result.regret == pytest.approx(100.0 - result.best_found_value)
    assert not set(result.selected_candidate_ids).intersection(split["test_ids"])


def test_prediction_intervals_use_shared_multiplier(monkeypatch, small_dataset):
    from backend.ml.models import bayesian

    monkeypatch.setattr(bayesian, "CI_MULTIPLIER", 2.5)
    X, y, ids = small_dataset
    split = _split_dataset(X, y, ids)
    config = ExperimentConfig(
        strategy="random", model_type="gp", initial_samples=12, batch_size=3,
        total_budget=30, random_seed=42, n_restarts_optimizer=0,
    )
    engine = ActiveLearningEngine(config)
    state = engine.initialize(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        split["test_X"], split["test_y"], split["test_ids"],
    )
    result, _ = engine.run_iteration(state)
    prediction = result.all_predictions[0]
    assert prediction.lower_cb == pytest.approx(
        prediction.predicted_mean - bayesian.CI_MULTIPLIER * prediction.predicted_std
    )
    assert prediction.upper_cb == pytest.approx(
        prediction.predicted_mean + bayesian.CI_MULTIPLIER * prediction.predicted_std
    )


def test_non_gp_ensemble_spread_is_not_exposed_as_95_percent_interval(small_dataset):
    X, y, ids = small_dataset
    split = _split_dataset(X, y, ids)
    engine = _make_engine("random", model_type="rf")
    state = engine.initialize(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        split["test_X"], split["test_y"], split["test_ids"],
    )
    result, _ = engine.run_iteration(state)
    assert result.all_predictions
    assert all(p.lower_cb is None and p.upper_cb is None for p in result.all_predictions)


@pytest.mark.parametrize("strategy", ["random", "uncertainty", "ei", "puda"])
def test_run_iteration_returns_result(strategy, small_dataset):
    X, y, ids = small_dataset
    split = _split_dataset(X, y, ids)
    engine = _make_engine(strategy)
    state = engine.initialize(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        split["test_X"], split["test_y"], split["test_ids"],
    )
    result, state = engine.run_iteration(state)
    assert result.iteration == 1
    assert len(result.selected_candidate_ids) == 5
    assert len(result.selected_true_values) == 5
    assert result.test_mae >= 0
    assert result.test_rmse >= 0
    assert result.test_r2 <= 1.0   # R² can be negative on tiny seed sets
    assert result.budget_used == 25     # 20 seed + 5 batch


def test_budget_respected(small_dataset):
    X, y, ids = small_dataset
    split = _split_dataset(X, y, ids)
    engine = _make_engine("random")
    state = engine.initialize(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        split["test_X"], split["test_y"], split["test_ids"],
    )
    # Run until budget exhausted
    for _ in range(10):
        if state.current_iteration >= 4:
            break
        _, state = engine.run_iteration(state)
    assert len(state.labeled_indices) <= engine.config.total_budget


def test_no_duplicate_candidates_across_iterations(small_dataset):
    X, y, ids = small_dataset
    split = _split_dataset(X, y, ids)
    engine = _make_engine("puda")
    state = engine.initialize(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        split["test_X"], split["test_y"], split["test_ids"],
    )
    result1, state = engine.run_iteration(state)
    result2, state = engine.run_iteration(state)
    iter1_ids = set(result1.selected_candidate_ids)
    iter2_ids = set(result2.selected_candidate_ids)
    assert len(iter1_ids & iter2_ids) == 0, "Same candidate must not be selected twice"


def test_labeled_set_grows_monotonically(small_dataset):
    X, y, ids = small_dataset
    split = _split_dataset(X, y, ids)
    engine = _make_engine("uncertainty")
    state = engine.initialize(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        split["test_X"], split["test_y"], split["test_ids"],
    )
    sizes = [len(state.labeled_indices)]
    for _ in range(3):
        _, state = engine.run_iteration(state)
        sizes.append(len(state.labeled_indices))
    assert sizes == sorted(sizes), "Labeled set must grow monotonically"


def test_get_summary_keys(small_dataset):
    X, y, ids = small_dataset
    split = _split_dataset(X, y, ids)
    engine = _make_engine("random")
    state = engine.initialize(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        split["test_X"], split["test_y"], split["test_ids"],
    )
    _, state = engine.run_iteration(state)
    summary = engine.get_summary(state)
    for key in ["experiment_id", "strategy", "best_found_value", "final_test_rmse",
                "budget_used", "rmse_curve", "best_found_curve", "regret_curve"]:
        assert key in summary
