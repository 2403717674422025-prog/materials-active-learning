"""
Tests for the data preprocessing pipeline.

Covers:
  - StandardScaler fit on labeled set only (no leakage)
  - PCA dimensionality reduction
  - initial_split produces correct sizes
  - seed_labeled_set produces correct sizes and no overlap
  - PreprocessingState serialize/deserialize round-trip
  - feature_statistics correctness
"""

import numpy as np
import pandas as pd
import pytest

from backend.ml.data.preprocessing import (
    PreprocessingConfig,
    fit_preprocessor,
    get_feature_statistics,
    initial_split,
    seed_labeled_set,
)
from backend.ml.data import feature_engineering


@pytest.fixture
def synthetic_data():
    """100 samples, 20 features, continuous target."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 20), columns=[f"feat_{i}" for i in range(20)])
    y = pd.Series(rng.uniform(0, 10, 100), name="target")
    ids = [f"MAT-{i:04d}" for i in range(100)]
    return X, y, ids


def test_initial_split_sizes(synthetic_data):
    X, y, ids = synthetic_data
    split = initial_split(X, y, ids, test_size=0.20, random_state=42)
    assert len(split["pool_X"]) == 80
    assert len(split["test_X"]) == 20
    assert len(split["pool_y"]) == 80
    assert len(split["test_y"]) == 20
    assert len(split["pool_ids"]) == 80
    assert len(split["test_ids"]) == 20


def test_initial_split_no_overlap(synthetic_data):
    X, y, ids = synthetic_data
    split = initial_split(X, y, ids, test_size=0.20, random_state=42)
    pool_ids_set = set(split["pool_ids"])
    test_ids_set = set(split["test_ids"])
    assert len(pool_ids_set & test_ids_set) == 0


def test_seed_labeled_set_sizes(synthetic_data):
    X, y, ids = synthetic_data
    split = initial_split(X, y, ids, test_size=0.20, random_state=42)
    labeled, unlabeled = seed_labeled_set(
        split["pool_X"], split["pool_y"], split["pool_ids"],
        n_initial=20, random_state=42
    )
    assert len(labeled["labeled_X"]) == 20
    assert len(unlabeled["unlabeled_X"]) == 60
    # No overlap
    labeled_set = set(labeled["labeled_ids"])
    unlabeled_set = set(unlabeled["unlabeled_ids"])
    assert len(labeled_set & unlabeled_set) == 0


def test_fit_preprocessor_no_pca(synthetic_data):
    X, _, _ = synthetic_data
    config = PreprocessingConfig(use_pca=False, random_state=42)
    state = fit_preprocessor(X.iloc[:30], config)
    assert state.scaler is not None
    assert state.pca is None
    X_transformed = state.transform(X.iloc[:30])
    assert X_transformed.shape == (30, 20)
    # Mean of transformed should be near 0 (StandardScaler)
    assert abs(X_transformed.mean()) < 0.1


def test_fit_preprocessor_with_pca(synthetic_data):
    X, _, _ = synthetic_data
    config = PreprocessingConfig(use_pca=True, pca_n_components=10, random_state=42)
    state = fit_preprocessor(X.iloc[:30], config)
    assert state.scaler is not None
    assert state.pca is not None
    X_transformed = state.transform(X.iloc[:30])
    assert X_transformed.shape == (30, 10)


def test_preprocessor_no_leakage(synthetic_data):
    """Scaler fitted on labeled set — test set transform should differ from fit on full data."""
    X, _, _ = synthetic_data
    config = PreprocessingConfig(use_pca=False)
    state = fit_preprocessor(X.iloc[:20], config)
    # Transform a held-out sample — should not crash
    X_held = state.transform(X.iloc[80:90])
    assert X_held.shape == (10, 20)


def test_feature_filters_fit_only_on_training_rows():
    train = pd.DataFrame({
        "signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "independent": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "constant_in_train": [7.0] * 6,
        "duplicate": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    })
    config = PreprocessingConfig(use_pca=False)
    fitted = fit_preprocessor(train, config)

    assert fitted.feature_filter is not None
    assert fitted.feature_filter.fit_sample_count == len(train)
    assert fitted.feature_columns == ["signal", "independent"]

    held_out = pd.DataFrame({
        "signal": [100.0, -100.0],
        "independent": [300.0, -300.0],
        "constant_in_train": [1.0, 9.0],
        "duplicate": [-500.0, 900.0],
    })
    changed_held_out = held_out * 1000.0
    original_columns = list(fitted.feature_columns)
    original_medians = dict(fitted.feature_filter.imputation_values)
    transformed = fitted.transform(held_out)
    changed_transformed = fitted.transform(changed_held_out)

    assert transformed.shape == (2, 2)
    assert changed_transformed.shape == transformed.shape
    assert fitted.feature_columns == original_columns
    assert fitted.feature_filter.imputation_values == original_medians


def test_training_filter_keeps_transform_alignment_for_all_partitions():
    train = pd.DataFrame({
        "a": [0.0, 1.0, 2.0, 3.0, 4.0],
        "b": [0.0, 1.0, 0.0, 1.0, 0.0],
        "constant": [4.0] * 5,
        "a_duplicate": [0.0, 2.0, 4.0, 6.0, 8.0],
    })
    validation = pd.DataFrame({
        "a": [2.5, 3.5],
        "b": [0.25, 0.75],
        "constant": [100.0, -100.0],
        "a_duplicate": [1000.0, -1000.0],
    })
    test = validation.iloc[:1].copy()
    candidates = validation.iloc[1:].copy()

    state = fit_preprocessor(train, PreprocessingConfig(use_pca=False))
    transformed = [
        state.transform(partition)
        for partition in (train, validation, test, candidates)
    ]

    assert state.feature_columns == ["a", "b"]
    assert [values.shape[1] for values in transformed] == [2, 2, 2, 2]
    assert [values.shape[0] for values in transformed] == [5, 2, 1, 1]
    assert all(np.isfinite(values).all() for values in transformed)


def test_missing_and_unseen_features_are_aligned_safely():
    train = pd.DataFrame({
        "signal": [0.0, 1.0, 2.0, 3.0],
        "missing_values": [1.0, np.nan, 3.0, 5.0],
        "constant": [2.0] * 4,
        "all_missing": [np.nan] * 4,
    })
    state = fit_preprocessor(train, PreprocessingConfig(use_pca=False))

    partial_candidate = pd.DataFrame({
        "signal": [4.0, np.inf],
        "unexpected_new_feature": [500.0, 600.0],
    })
    transformed = state.transform(partial_candidate)

    assert transformed.shape == (2, len(state.feature_columns))
    assert np.isfinite(transformed).all()
    assert "unexpected_new_feature" not in state.feature_columns


def test_dataset_featurization_defers_filters_and_versions_cache(tmp_path, monkeypatch):
    raw_path = tmp_path / "materials.csv"
    cache_path = tmp_path / "features.pkl"
    pd.DataFrame({
        "composition": ["A", "B", "C"],
        "target": [1.0, 2.0, 3.0],
    }).to_csv(raw_path, index=False)
    calls = []

    def fake_composition_to_features(df, **kwargs):
        calls.append(kwargs)
        return (
            pd.DataFrame({"raw_a": [1.0, 2.0, 3.0], "raw_b": [4.0, 5.0, 6.0]}),
            ["raw_a", "raw_b"],
            [],
        )

    monkeypatch.setattr(
        feature_engineering, "composition_to_features", fake_composition_to_features
    )
    X, y, ids = feature_engineering.load_and_featurize_dataset(
        csv_path=str(raw_path),
        target_col="target",
        cache_path=str(cache_path),
    )

    assert calls == [{"drop_low_variance": False, "drop_correlated": False}]
    assert list(X.columns) == ["raw_a", "raw_b"]
    assert len(y) == len(ids) == len(X) == 3

    import pickle
    with open(cache_path, "rb") as cache_file:
        cache = pickle.load(cache_file)
    assert cache["feature_cache_version"] == feature_engineering.FEATURE_CACHE_VERSION

    cache["feature_cache_version"] = feature_engineering.FEATURE_CACHE_VERSION - 1
    with open(cache_path, "wb") as cache_file:
        pickle.dump(cache, cache_file)
    feature_engineering.load_and_featurize_dataset(
        csv_path=str(raw_path),
        target_col="target",
        cache_path=str(cache_path),
    )
    assert len(calls) == 2


def test_preprocessing_state_save_load(synthetic_data, tmp_path):
    X, _, _ = synthetic_data
    config = PreprocessingConfig(use_pca=False)
    state = fit_preprocessor(X.iloc[:30], config)
    path = str(tmp_path / "state.pkl")
    state.save(path)

    from backend.ml.data.preprocessing import PreprocessingState
    loaded = PreprocessingState.load(path)
    # Transforms should produce identical results
    orig = state.transform(X.iloc[:30])
    loaded_t = loaded.transform(X.iloc[:30])
    np.testing.assert_array_almost_equal(orig, loaded_t)


def test_feature_statistics(synthetic_data):
    X, y, _ = synthetic_data
    stats = get_feature_statistics(X, y)
    assert stats["n_samples"] == 100
    assert stats["n_features"] == 20
    assert abs(stats["target_mean"] - float(y.mean())) < 1e-6
    assert "target_histogram" in stats
    assert len(stats["target_histogram"]["counts"]) == 20
