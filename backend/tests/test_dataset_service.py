"""Regression tests for dataset cache schema changes."""

import pickle

import pandas as pd

from backend.app.models.dataset import Dataset
from backend.app.services import dataset_service
from backend.ml.data import feature_engineering


def test_builtin_dataset_reloads_legacy_cache_without_mutating_history(
    tmp_path, monkeypatch
):
    legacy_cache_path = tmp_path / "legacy.pkl"
    with open(legacy_cache_path, "wb") as cache_file:
        pickle.dump({
            "X": pd.DataFrame({"old_feature": [1.0, 2.0, 3.0]}),
            "y": pd.Series([1.0, 2.0, 3.0]),
            "candidate_ids": ["A", "B", "C"],
            "feature_cache_version": feature_engineering.FEATURE_CACHE_VERSION - 1,
        }, cache_file)

    legacy_record = Dataset(
        id="legacy-dataset",
        name="matbench_expt_gap",
        source="matbench_expt_gap",
        target_column="gap expt",
        status="ready",
        num_samples=3,
        num_features=1,
        cache_path=str(legacy_cache_path),
    )

    class Query:
        def __init__(self, records):
            self.records = records

        def filter(self, *_args):
            return self

        def all(self):
            return self.records

    class Session:
        def __init__(self):
            self.records = [legacy_record]

        def query(self, _model):
            return Query(self.records)

        def add(self, record):
            self.records.append(record)

        def commit(self):
            pass

    session = Session()
    monkeypatch.setattr(
        dataset_service,
        "settings",
        type("Settings", (), {"dataset_cache_dir": str(tmp_path)})(),
    )
    calls = []

    def fake_load_and_featurize_dataset(*, csv_path, target_col, cache_path):
        calls.append(cache_path)
        X = pd.DataFrame({"new_feature": [1.0, 2.0, 3.0]})
        y = pd.Series([1.0, 2.0, 3.0], name=target_col)
        candidate_ids = ["A", "B", "C"]
        with open(cache_path, "wb") as cache_file:
            pickle.dump({
                "X": X,
                "y": y,
                "candidate_ids": candidate_ids,
                "feature_cache_version": feature_engineering.FEATURE_CACHE_VERSION,
            }, cache_file)
        return X, y, candidate_ids

    monkeypatch.setattr(
        feature_engineering,
        "load_and_featurize_dataset",
        fake_load_and_featurize_dataset,
    )

    current_record = dataset_service.load_builtin_dataset(session)

    assert current_record is not legacy_record
    assert current_record.status == "ready"
    assert current_record.feature_columns == ["new_feature"]
    assert current_record.preprocessing_config == {
        "feature_cache_version": feature_engineering.FEATURE_CACHE_VERSION
    }
    assert legacy_record.status == "ready"
    assert legacy_record.cache_path == str(legacy_cache_path)
    assert calls == [current_record.cache_path]

    reused_record = dataset_service.load_builtin_dataset(session)
    assert reused_record is current_record
    assert calls == [current_record.cache_path]
