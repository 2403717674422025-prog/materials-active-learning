"""Tests for leakage-safe prediction/observation calibration matching."""

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database import Base
from backend.app.models.dataset import Dataset
from backend.app.models.experiment import Experiment
from backend.app.models.result import CandidatePredictionRecord, ExperimentalResult
from backend.app.services import benchmark_service
from backend.app.services.benchmark_service import get_calibration_metrics


@pytest.fixture
def calibration_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        dataset_id = str(uuid.uuid4())
        db.add(Dataset(
            id=dataset_id, name="calibration-test", status="ready",
            global_maximum=100.0,
        ))
        db.flush()
        experiment_id = str(uuid.uuid4())
        other_experiment_id = str(uuid.uuid4())
        db.add_all([
            Experiment(id=experiment_id, dataset_id=dataset_id, current_iteration=4),
            Experiment(id=other_experiment_id, dataset_id=dataset_id, current_iteration=4),
        ])
        db.commit()
        yield db, experiment_id, other_experiment_id
    Base.metadata.drop_all(engine)
    engine.dispose()


def _prediction(exp_id, candidate_id, iteration, mean, std=1.0, record_id=None):
    return CandidatePredictionRecord(
        id=record_id or str(uuid.uuid4()), experiment_id=exp_id,
        iteration_num=iteration, candidate_id=candidate_id,
        predicted_mean=mean, predicted_std=std,
    )


def _observation(exp_id, candidate_id, iteration, value):
    return ExperimentalResult(
        id=str(uuid.uuid4()), experiment_id=exp_id,
        iteration_num=iteration, candidate_id=candidate_id, true_value=value,
    )


def test_calibration_uses_latest_pre_acquisition_prediction_only(
    calibration_db, monkeypatch,
):
    db, exp_id, other_exp_id = calibration_db
    truths = {f"valid-{i}": float(10 + i) for i in range(5)}
    for candidate_id, truth in truths.items():
        db.add_all([
            _observation(exp_id, candidate_id, 2, truth),
            _prediction(exp_id, candidate_id, 1, truth + 2),
            _prediction(exp_id, candidate_id, 2, truth + 1),
            _prediction(exp_id, candidate_id, 3, truth + 99),
            # Same candidate ID in another experiment must never be joined.
            _prediction(other_exp_id, candidate_id, 2, truth - 99),
        ])

    db.add_all([
        _observation(exp_id, "after-only", 2, 4.0),
        _prediction(exp_id, "after-only", 3, 4.0),
        _observation(exp_id, "no-prediction", 2, 5.0),
        _observation(exp_id, "missing-std", 2, 6.0),
        _prediction(exp_id, "missing-std", 2, 6.0, std=None),
        _observation(exp_id, "missing-mean", 2, 6.5),
        _prediction(exp_id, "missing-mean", 2, None),
        _observation(exp_id, "zero-std", 2, 7.0),
        _prediction(exp_id, "zero-std", 2, 7.0, std=0.0),
        _observation(exp_id, "negative-std", 2, 8.0),
        _prediction(exp_id, "negative-std", 2, 8.0, std=-1.0),
        _observation(exp_id, "infinite-mean", 2, 9.0),
        _prediction(exp_id, "infinite-mean", 2, float("inf")),
        _observation(exp_id, "infinite-std", 2, 10.0),
        _prediction(exp_id, "infinite-std", 2, 10.0, std=float("inf")),
        _observation(exp_id, "infinite-true", 2, float("inf")),
        _prediction(exp_id, "infinite-true", 2, 11.0),
    ])
    db.commit()

    def hidden_dataset_access(*args, **kwargs):
        raise AssertionError("Calibration must not read unobserved dataset targets")

    monkeypatch.setattr(benchmark_service, "load_dataset_arrays", hidden_dataset_access)
    result = get_calibration_metrics(db, exp_id)

    assert result["n_aligned"] == 5
    assert result["n_observations"] == 5
    assert result["n_excluded"] == 9
    assert result["exclusion_counts"] == {
        "prediction_after_acquisition": 1,
        "no_prediction_record": 1,
        "missing_predicted_std": 1,
        "non_positive_predicted_std": 2,
        "missing_or_invalid_predicted_mean": 2,
        "invalid_predicted_std": 1,
        "invalid_true_value": 1,
    }
    assert result["rmse_on_observed"] == pytest.approx(1.0)
    assert result["mae_on_observed"] == pytest.approx(1.0)
    assert result["coverage_90pct"] == pytest.approx(1.0)
    assert result["coverage_95pct"] == pytest.approx(1.0)
    assert len(result["reliability_diagram"]["expected_coverage"]) == len(
        result["reliability_diagram"]["observed_coverage"]
    )


def test_calibration_returns_insufficient_data_counts(calibration_db):
    db, exp_id, _ = calibration_db
    for i in range(4):
        candidate_id = f"few-{i}"
        db.add_all([
            _observation(exp_id, candidate_id, 1, float(i)),
            _prediction(exp_id, candidate_id, 1, float(i)),
        ])
    db.commit()

    result = get_calibration_metrics(db, exp_id)
    assert "error" in result
    assert result["n_aligned"] == 4
    assert result["n_excluded"] == 0
    assert result["minimum_required"] == 5
    assert "ece" not in result
