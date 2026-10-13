"""Regression tests for budget-aligned benchmark response curves."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.models.benchmark import BenchmarkRun
from backend.app.models.dataset import Dataset
from backend.app.models.experiment import Experiment
from backend.database import Base
from backend.app.services import benchmark_service
from backend.app.services.benchmark_service import (
    _build_summary,
    get_benchmark_run,
    recover_interrupted_benchmarks,
    run_strategy_benchmark,
)
from backend.ml.evaluation.metrics import normalized_area_under_discovery_curve


def _run(discovery_budgets, best, iteration_budgets, rmse):
    return {
        "strategy": "puda",
        "status": "completed",
        "discovery_budget_curve": discovery_budgets,
        "best_found_curve": best,
        "regret_curve": [10.0 - value for value in best],
        "budget_curve": iteration_budgets,
        "acquisition_budget_curve": [budget + 1 for budget in iteration_budgets],
        "rmse_curve": rmse,
        "mae_curve": rmse,
        "diversity_curve": [0.2] * len(iteration_budgets),
        "iteration_runtime_curve": [0.1] * len(iteration_budgets),
        "naudc": 0.5,
        "top10_discovery_rate": 0.0,
        "final_mae": rmse[-1] if rmse else None,
        "final_rmse": rmse[-1] if rmse else None,
        "final_regret": 10.0 - best[-1],
        "global_maximum": 10.0,
        "actual_budget": iteration_budgets[-1] if iteration_budgets else 5,
        "runtime_seconds": 0.2,
    }


def test_summary_uses_each_curve_own_budget_axis_with_partial_batch():
    # Seed budget 7, then a batch of 3 and a final partial batch of 1.
    result = _build_summary([
        _run([7, 8, 9, 10, 11], [2, 2, 4, 4, 8], [10, 11], [3, 2]),
    ], [1])

    discovery = result["discovery_curve"][0]
    regret = result["regret_curve"][0]
    rmse = result["rmse_curve"][0]
    assert discovery["budget"] == [7, 8, 9, 10, 11]
    assert discovery["best_found"] == [2, 2, 4, 4, 8]
    assert regret["budget"] == discovery["budget"]
    assert len(rmse["budget"]) == len(rmse["rmse"]) == 2
    assert rmse["budget"] == [10, 11]
    assert discovery["budget_unit"] == "cumulative_candidates"


def test_summary_keeps_mismatched_run_lengths_keyed_to_actual_budgets():
    result = _build_summary([
        _run([5, 6, 7], [1, 2, 3], [7], [4]),
        _run([5, 6], [3, 4], [6], [2]),
    ], [1, 2])
    discovery = result["discovery_curve"][0]
    assert discovery["budget"] == [5, 6, 7]
    assert discovery["best_found"] == [2.0, 3.0, 3.0]
    assert result["rmse_curve"][0]["budget"] == [6, 7]
    assert len(result["rmse_curve"][0]["budget"]) == len(result["rmse_curve"][0]["rmse"])


def test_runner_labels_non_default_and_partial_batches_at_candidate_counts(monkeypatch):
    class FakeEngine:
        def __init__(self, config):
            self.config = config
            self.steps = iter([([2.0, 4.0, 1.0], 13), ([5.0], 14)])

        def initialize(self, *args, **kwargs):
            self.initialization = kwargs
            return SimpleNamespace(
                acquired_values={i: float(i) for i in range(10)},
                labeled_indices=list(range(10)), iterations=[],
            )

        def run_iteration(self, state):
            selected, budget = next(self.steps)
            state.labeled_indices.extend(range(len(state.labeled_indices), budget))
            state.iterations.append(object())
            return SimpleNamespace(
                selected_true_values=selected, budget_used=budget,
                selected_candidate_ids=[str(i) for i in range(len(selected))],
                test_rmse=1.0, test_mae=0.5,
                diversity_score=0.25, iteration_time_seconds=0.1,
            ), state

        def train_model(self, state):
            class Model:
                @staticmethod
                def evaluate(_X, _y):
                    return {"mae": 0.25, "rmse": 0.5}

            return Model(), SimpleNamespace(transform=lambda X: X.to_numpy())

    monkeypatch.setattr(benchmark_service, "ActiveLearningEngine", FakeEngine)
    result = benchmark_service._run_single_experiment(
        pd.DataFrame({"x": np.arange(20)}), pd.Series(np.arange(20, dtype=float)),
        [str(i) for i in range(20)], pd.DataFrame({"x": [20]}), pd.Series([21.0]), ["test"],
        strategy="random", model_type="rf", initial_samples=10, batch_size=3,
        total_budget=14, random_seed=1, seed_indices=list(range(10)),
    )

    assert result["budget_curve"] == [10, 13]
    assert result["acquisition_budget_curve"] == [13, 14]
    assert result["discovery_budget_curve"] == [10, 11, 12, 13, 14]
    assert len(result["best_found_curve"]) == len(result["discovery_budget_curve"])
    assert result["final_mae"] == 0.25
    assert result["final_rmse"] == 0.5
    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["naudc"] == normalized_area_under_discovery_curve(
        result["best_found_curve"], result["global_maximum"],
        budgets=result["discovery_budget_curve"],
    )


def test_summary_uses_sample_standard_deviation_and_reports_counts():
    first = _run([10, 11], [2, 4], [10], [3])
    second = _run([10, 11], [4, 8], [10], [1])
    summary = _build_summary([first, second], [42, 123], ["puda"])
    row = summary["summary_table"][0]

    assert row["naudc_std"] == pytest.approx(
        np.std([first["naudc"], second["naudc"]], ddof=1)
    )
    assert summary["discovery_curve"][0]["n_runs"] == [2, 2]

    single = _build_summary([first], [42], ["puda"])["summary_table"][0]
    assert single["n_seeds"] == 1
    assert single["naudc_std"] is None


def test_summary_aggregates_curves_and_runtime_from_completed_runs_only():
    first = _run([10, 11], [2, 4], [10], [3])
    second = _run([10, 11], [4, 8], [10], [1])
    first["runtime_seconds"] = 2.0
    second["runtime_seconds"] = 4.0
    incomplete = _run([10, 11, 12], [100, 100, 100], [10, 12], [100, 100])
    incomplete.update(status="incomplete", runtime_seconds=100.0)
    failed = _run([99], [999], [99], [999])
    failed.update(status="failed", runtime_seconds=200.0)

    summary = _build_summary(
        [first, second, incomplete, failed], [42, 123, 456, 789], ["puda"]
    )
    row = summary["summary_table"][0]
    discovery = summary["discovery_curve"][0]
    rmse = summary["rmse_curve"][0]

    assert discovery["budget"] == [10, 11]
    assert discovery["best_found"] == [3.0, 6.0]
    assert discovery["best_found_std"] == pytest.approx(
        [np.std([2.0, 4.0], ddof=1), np.std([4.0, 8.0], ddof=1)]
    )
    assert discovery["n_runs"] == [2, 2]
    assert rmse["budget"] == [10]
    assert rmse["rmse"] == [2.0]
    assert rmse["n_runs"] == [2]
    assert row["runtime_seconds_mean"] == 3.0
    assert row["runtime_seconds_std"] == pytest.approx(np.std([2.0, 4.0], ddof=1))
    assert row["n_seeds"] == 2
    assert row["n_incomplete"] == 1
    assert row["n_failed"] == 1

    one_completed = _build_summary(
        [first, incomplete, failed], [42, 123, 456], ["puda"]
    )
    assert one_completed["discovery_curve"][0]["n_runs"] == [1, 1]
    assert one_completed["discovery_curve"][0]["best_found_std"] == [None, None]
    assert one_completed["summary_table"][0]["runtime_seconds_std"] is None


def test_initial_candidate_indices_are_seed_reproducible_and_distinct():
    first = benchmark_service._initial_indices(50, 10, 42)
    repeat = benchmark_service._initial_indices(50, 10, 42)
    another_seed = benchmark_service._initial_indices(50, 10, 123)

    assert first == repeat
    assert first != another_seed
    assert len(set(first)) == 10


def test_dataset_test_targets_do_not_define_discovery_reference_or_seed(monkeypatch):
    captured = []

    class FakeEngine:
        def __init__(self, config):
            self.config = config

        def initialize(self, *args, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                acquired_values={index: float(index) for index in [1, 3]},
                labeled_indices=[1, 3],
            )

        def train_model(self, _state):
            class Model:
                @staticmethod
                def evaluate(_X, y):
                    return {"mae": float(np.mean(y)), "rmse": float(np.mean(y))}
            return Model(), SimpleNamespace(transform=lambda X: X.to_numpy())

        def run_iteration(self, _state):
            raise RuntimeError("budget exhausted")

    monkeypatch.setattr(benchmark_service, "ActiveLearningEngine", FakeEngine)
    args = (
        pd.DataFrame({"x": np.arange(8)}),
        pd.Series(np.arange(8, dtype=float)),
        [str(i) for i in range(8)],
        pd.DataFrame({"x": [10, 11]}),
    )
    first = benchmark_service._run_single_experiment(
        *args, pd.Series([8.0, 9.0]), ["t1", "t2"],
        strategy="random", model_type="rf", initial_samples=2,
        batch_size=1, total_budget=4, random_seed=42, seed_indices=[1, 3],
    )
    second = benchmark_service._run_single_experiment(
        *args, pd.Series([800.0, 900.0]), ["t1", "t2"],
        strategy="random", model_type="rf", initial_samples=2,
        batch_size=1, total_budget=4, random_seed=42, seed_indices=[1, 3],
    )

    assert first["initial_candidate_ids"] == second["initial_candidate_ids"]
    assert first["acquired_candidate_ids"] == second["acquired_candidate_ids"] == []
    assert first["global_maximum"] == second["global_maximum"] == 7.0
    assert captured[0]["dataset_maximum"] == captured[1]["dataset_maximum"] == 7.0
    assert first["final_rmse"] != second["final_rmse"]


def test_benchmark_results_are_persisted_and_retrievable(monkeypatch):
    dataset = Dataset(
        id="dataset-id", name="synthetic", status="ready",
        global_maximum=9.0,
    )
    experiment = Experiment(
        id="experiment-id", dataset_id=dataset.id, random_seed=42,
        initial_samples=2, batch_size=1, total_budget=4,
        xi=0.01, kappa=2.0, alpha=0.4, beta=0.35, gamma=0.25,
    )
    X = pd.DataFrame({"feature": np.arange(20, dtype=float)})
    y = pd.Series(np.arange(20, dtype=float))
    ids = [f"c-{index}" for index in range(20)]
    records = []

    class Query:
        def __init__(self, model):
            self.model = model

        def filter(self, *_conditions):
            return self

        def first(self):
            return experiment if self.model is Experiment else records[-1]

    class FakeDB:
        def __init__(self):
            self.commit_count = 0

        def query(self, model):
            return Query(model)

        def add(self, record):
            records.append(record)

        def commit(self):
            self.commit_count += 1

    db = FakeDB()
    monkeypatch.setattr(benchmark_service, "get_dataset_by_id", lambda *_: dataset)
    monkeypatch.setattr(
        benchmark_service, "load_dataset_arrays", lambda *_: (X, y, ids)
    )

    def fake_run(*args, strategy, random_seed, seed_indices, **_kwargs):
        if strategy == "ei" and random_seed == 123:
            raise RuntimeError("synthetic model failure")
        pool_ids = args[2]
        return {
            **_run([len(seed_indices), len(seed_indices) + 2], [1, 3], [len(seed_indices)], [1]),
            "strategy": strategy,
            "seed": random_seed,
            "initial_candidate_ids": [pool_ids[index] for index in seed_indices],
            "status": (
                "completed" if random_seed == 42 else "incomplete"
            ),
            "actual_budget": len(seed_indices) + 2,
        }

    monkeypatch.setattr(benchmark_service, "_run_single_experiment", fake_run)
    summary = run_strategy_benchmark(
        db, experiment.id, strategies=["random", "ei"], seeds=[42, 123],
        model_type="rf",
    )

    persisted = records[0]
    assert isinstance(persisted, BenchmarkRun)
    assert summary["benchmark_id"] == persisted.id
    assert persisted.status == "partial"
    assert len(persisted.results) == 4
    assert [result["status"] for result in persisted.results].count("incomplete") == 1
    assert [result["status"] for result in persisted.results].count("failed") == 1
    assert len(summary["failed_or_incomplete_runs"]) == 2
    failed_result = next(
        result for result in persisted.results if result["status"] == "failed"
    )
    assert failed_result["error"] == "RuntimeError: synthetic model failure"
    assert len(failed_result["initial_candidate_ids"]) == 2
    assert db.commit_count >= len(persisted.results) + 2
    assert persisted.configuration["split"]["test_candidate_ids"]
    assert persisted.configuration["dataset_fingerprint"]
    initial_by_seed = {}
    for result in persisted.results:
        initial_by_seed.setdefault(result["seed"], {})[result["strategy"]] = (
            tuple(result["initial_candidate_ids"])
        )
    assert all(len(set(by_strategy.values())) == 1 for by_strategy in initial_by_seed.values())
    assert persisted.configuration["model"]["n_estimators"] == 100

    restored = get_benchmark_run(db, experiment.id, persisted.id)
    assert restored["results"] == persisted.results
    assert restored["summary"]["summary_table"] == summary["summary_table"]


def test_incomplete_run_reports_actual_partial_budget_and_error(monkeypatch):
    class FakeEngine:
        def __init__(self, _config):
            self.iterations = 0

        def initialize(self, *_args, **kwargs):
            indices = kwargs["seed_indices"]
            return SimpleNamespace(
                acquired_values={index: float(index) for index in indices},
                labeled_indices=list(indices),
            )

        def run_iteration(self, state):
            self.iterations += 1
            if self.iterations > 1:
                raise RuntimeError("synthetic acquisition stopped")
            state.labeled_indices.extend([4, 5])
            return SimpleNamespace(
                selected_true_values=[4.0, 5.0],
                selected_candidate_ids=["c4", "c5"],
                test_rmse=2.0,
                test_mae=1.0,
                diversity_score=0.3,
                iteration_time_seconds=0.1,
            ), state

        @staticmethod
        def train_model(_state):
            class Model:
                @staticmethod
                def evaluate(_X, _y):
                    return {"mae": 0.5, "rmse": 0.75}

            return Model(), SimpleNamespace(transform=lambda X: X.to_numpy())

    monkeypatch.setattr(benchmark_service, "ActiveLearningEngine", FakeEngine)
    result = benchmark_service._run_single_experiment(
        pd.DataFrame({"x": np.arange(8)}),
        pd.Series(np.arange(8, dtype=float)),
        [f"c{i}" for i in range(8)],
        pd.DataFrame({"x": [8.0]}),
        pd.Series([9.0]),
        ["test"],
        strategy="puda",
        model_type="rf",
        initial_samples=2,
        batch_size=2,
        total_budget=5,
        random_seed=42,
        seed_indices=[0, 1],
    )
    assert result["status"] == "incomplete"
    assert result["actual_budget"] == 4
    assert result["requested_budget"] == 5
    assert result["discovery_budget_curve"] == [2, 3, 4]
    assert result["error"] == "synthetic acquisition stopped"


def test_benchmark_run_json_round_trips_through_sqlite(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'benchmark.sqlite'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        dataset = Dataset(id="persist-dataset", name="fixture", status="ready")
        session.add(dataset)
        session.commit()
        experiment = Experiment(
            id="persist-experiment",
            dataset_id=dataset.id,
            initial_samples=2,
            batch_size=1,
            total_budget=4,
        )
        session.add(experiment)
        session.commit()
        run = BenchmarkRun(
            id="persist-benchmark",
            experiment_id=experiment.id,
            status="partial",
            model_type="rf",
            configuration={"seed": 42, "split": {"pool": ["a", "b"]}},
            results=[{"strategy": "random", "status": "incomplete"}],
            summary={"naudc_mean": 0.5},
        )
        session.add(run)
        session.commit()

        loaded = session.get(BenchmarkRun, run.id)
        assert loaded is not None
        assert loaded.configuration == {"seed": 42, "split": {"pool": ["a", "b"]}}
        assert loaded.results == [{"strategy": "random", "status": "incomplete"}]
        assert loaded.summary == {"naudc_mean": 0.5}
    engine.dispose()


def test_recovery_marks_stale_runs_and_preserves_partial_results(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'recovery.sqlite'}")
    Base.metadata.create_all(engine)
    now = datetime(2025, 1, 2, 12, 0, 0)
    stale_time = now - timedelta(hours=2)
    fresh_time = now - timedelta(minutes=10)
    preserved_result = {
        "strategy": "random",
        "seed": 42,
        "status": "completed",
        "actual_budget": 4,
    }
    with Session(engine) as session:
        dataset = Dataset(id="recovery-dataset", name="fixture", status="ready")
        session.add(dataset)
        session.commit()
        experiment = Experiment(
            id="recovery-experiment",
            dataset_id=dataset.id,
            initial_samples=2,
            batch_size=1,
            total_budget=4,
        )
        session.add(experiment)
        session.commit()

        partial = BenchmarkRun(
            id="stale-partial",
            experiment_id=experiment.id,
            status="running",
            model_type="rf",
            configuration={
                "heartbeat_at": stale_time.isoformat(),
                "active_run": {
                    "strategy": "ei",
                    "seed": 123,
                    "initial_candidate_ids": ["c1", "c2"],
                    "requested_budget": 4,
                    "actual_budget": 3,
                },
            },
            results=[preserved_result],
            started_at=stale_time,
        )
        empty = BenchmarkRun(
            id="stale-empty",
            experiment_id=experiment.id,
            status="running",
            model_type="rf",
            configuration={"heartbeat_at": stale_time.isoformat()},
            results=[],
            started_at=stale_time,
        )
        fresh = BenchmarkRun(
            id="fresh-running",
            experiment_id=experiment.id,
            status="running",
            model_type="rf",
            configuration={"heartbeat_at": fresh_time.isoformat()},
            results=[],
            started_at=stale_time,
        )
        active = BenchmarkRun(
            id="active-running",
            experiment_id=experiment.id,
            status="running",
            model_type="rf",
            configuration={"heartbeat_at": stale_time.isoformat()},
            results=[],
            started_at=stale_time,
        )
        session.add_all([partial, empty, fresh, active])
        session.commit()

        monkeypatch.setattr(
            benchmark_service, "_ACTIVE_BENCHMARK_RUN_IDS", {"active-running"}
        )
        recovered = recover_interrupted_benchmarks(
            session, now=now, stale_after=timedelta(hours=1)
        )

        assert recovered == 2
        session.expire_all()
        partial = session.get(BenchmarkRun, "stale-partial")
        empty = session.get(BenchmarkRun, "stale-empty")
        fresh = session.get(BenchmarkRun, "fresh-running")
        active = session.get(BenchmarkRun, "active-running")
        assert partial is not None and empty is not None
        assert partial.status == "partial"
        assert partial.results[0] == preserved_result
        assert partial.results[1]["status"] == "failed"
        assert partial.results[1]["strategy"] == "ei"
        assert partial.finished_at == now
        assert "not rerun" in partial.error_summary
        assert partial.summary["interrupted"] is True
        assert partial.summary["failed_or_incomplete_runs"][0]["status"] == "failed"
        assert partial.summary["run_status_counts"] == {
            "completed": 1,
            "incomplete": 0,
            "failed": 1,
        }
        assert empty.status == "failed"
        assert empty.finished_at == now
        assert fresh is not None and fresh.status == "running"
        assert active is not None and active.status == "running"
    engine.dispose()
