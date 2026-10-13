"""
Tests for surrogate model implementations.

Covers:
  - RandomForestSurrogate: fit, predict shape, uncertainty non-negative
  - GaussianProcessSurrogate: fit, predict shape, true posterior std
  - Both models: evaluate() returns correct keys
  - Model save/load round-trip
  - Prediction std > 0 for unseen points (GP)
  - Prediction std ≥ 0 everywhere (no negative uncertainty)
"""

import numpy as np
import pytest

from backend.ml.models.baseline import RandomForestSurrogate
from backend.ml.models.bayesian import GaussianProcessSurrogate


@pytest.fixture
def simple_regression_data():
    """Simple synthetic regression data: y = sin(x) + noise."""
    rng = np.random.RandomState(1)
    X_train = rng.randn(50, 5)
    y_train = np.sin(X_train[:, 0]) + 0.1 * rng.randn(50)
    X_test = rng.randn(20, 5)
    y_test = np.sin(X_test[:, 0]) + 0.1 * rng.randn(20)
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomForestSurrogate:
    def test_fit_and_predict_shape(self, simple_regression_data):
        X_train, y_train, X_test, _ = simple_regression_data
        model = RandomForestSurrogate(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        mean, std = model.predict(X_test)
        assert mean.shape == (len(X_test),)
        assert std.shape == (len(X_test),)

    def test_uncertainty_non_negative(self, simple_regression_data):
        X_train, y_train, X_test, _ = simple_regression_data
        model = RandomForestSurrogate(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        _, std = model.predict(X_test)
        assert np.all(std >= 0), "Uncertainty must be non-negative"

    def test_evaluate_returns_correct_keys(self, simple_regression_data):
        X_train, y_train, X_test, y_test = simple_regression_data
        model = RandomForestSurrogate(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_not_fitted_raises(self, simple_regression_data):
        _, _, X_test, _ = simple_regression_data
        model = RandomForestSurrogate()
        with pytest.raises(RuntimeError):
            model.predict(X_test)

    def test_save_load_roundtrip(self, simple_regression_data, tmp_path):
        X_train, y_train, X_test, _ = simple_regression_data
        model = RandomForestSurrogate(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        path = str(tmp_path / "rf_model.pkl")
        model.save(path)
        loaded = RandomForestSurrogate.load(path)
        mean1, std1 = model.predict(X_test)
        mean2, std2 = loaded.predict(X_test)
        np.testing.assert_array_almost_equal(mean1, mean2)
        np.testing.assert_array_almost_equal(std1, std2)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian Process Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGaussianProcessSurrogate:
    def test_fit_and_predict_shape(self, simple_regression_data):
        X_train, y_train, X_test, _ = simple_regression_data
        model = GaussianProcessSurrogate(n_restarts_optimizer=1, random_state=0)
        model.fit(X_train, y_train)
        mean, std = model.predict(X_test)
        assert mean.shape == (len(X_test),)
        assert std.shape == (len(X_test),)

    def test_uncertainty_non_negative(self, simple_regression_data):
        X_train, y_train, X_test, _ = simple_regression_data
        model = GaussianProcessSurrogate(n_restarts_optimizer=1, random_state=0)
        model.fit(X_train, y_train)
        _, std = model.predict(X_test)
        assert np.all(std >= 0), "GP posterior std must be non-negative"

    def test_uncertainty_higher_for_unseen_regions(self, simple_regression_data):
        """GP should be more uncertain far from training data."""
        X_train, y_train, _, _ = simple_regression_data
        model = GaussianProcessSurrogate(n_restarts_optimizer=1, random_state=0)
        model.fit(X_train, y_train)
        # Points near training data
        X_near = X_train[:5] + np.random.RandomState(0).randn(5, 5) * 0.01
        # Points far from training data
        X_far = np.random.RandomState(0).randn(5, 5) * 10
        _, std_near = model.predict(X_near)
        _, std_far = model.predict(X_far)
        assert std_far.mean() > std_near.mean(), \
            "GP should be more uncertain far from training data"

    def test_predict_with_confidence_keys(self, simple_regression_data):
        X_train, y_train, X_test, _ = simple_regression_data
        model = GaussianProcessSurrogate(n_restarts_optimizer=1, random_state=0)
        model.fit(X_train, y_train)
        out = model.predict_with_confidence(X_test)
        assert "predicted_property" in out
        assert "prediction_std" in out
        assert "lower_confidence_bound" in out
        assert "upper_confidence_bound" in out
        # CI should bracket the mean
        assert np.all(out["lower_confidence_bound"] <= out["predicted_property"])
        assert np.all(out["upper_confidence_bound"] >= out["predicted_property"])

    def test_evaluate_returns_correct_keys(self, simple_regression_data):
        X_train, y_train, X_test, y_test = simple_regression_data
        model = GaussianProcessSurrogate(n_restarts_optimizer=1, random_state=0)
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        assert all(k in metrics for k in ["mae", "rmse", "r2"])

    def test_n_train_attribute(self, simple_regression_data):
        X_train, y_train, _, _ = simple_regression_data
        model = GaussianProcessSurrogate(n_restarts_optimizer=1, random_state=0)
        model.fit(X_train, y_train)
        assert model.n_train == len(X_train)
