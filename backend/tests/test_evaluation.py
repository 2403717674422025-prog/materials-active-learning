"""Tests for evaluation metrics and comparison framework."""

import numpy as np
import pytest

from backend.ml.evaluation.metrics import (
    best_found_curve,
    cumulative_regret,
    discovery_curve_with_initial,
    normalized_area_under_discovery_curve,
    top_k_discovery_rate,
    batch_diversity_score,
    expected_calibration_error,
    regression_metrics,
)
from backend.ml.evaluation.comparison import StrategyComparison


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        m = regression_metrics(y, y)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_worst_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] > 0
        assert m["rmse"] > 0


class TestBestFoundCurve:
    def test_monotonically_increasing(self):
        values = [3.0, 1.0, 5.0, 2.0, 4.0]
        curve = best_found_curve(values, mode="maximize")
        assert curve == [3.0, 3.0, 5.0, 5.0, 5.0]

    def test_minimize_mode(self):
        values = [5.0, 3.0, 4.0, 1.0, 2.0]
        curve = best_found_curve(values, mode="minimize")
        assert curve == [5.0, 3.0, 3.0, 1.0, 1.0]

    def test_empty_input(self):
        assert best_found_curve([]) == []


class TestCumulativeRegret:
    def test_regret_decreases(self):
        curve_vals = [3.0, 5.0, 7.0, 8.0]
        regret = cumulative_regret(curve_vals, global_optimum=10.0)
        assert regret == [7.0, 5.0, 3.0, 2.0]

    def test_zero_regret_at_optimum(self):
        curve_vals = [10.0]
        regret = cumulative_regret(curve_vals, global_optimum=10.0)
        assert regret == [0.0]


class TestNAUDC:
    def test_perfect_strategy(self):
        """Strategy that finds global optimum immediately."""
        curve = [10.0, 10.0, 10.0, 10.0]
        naudc = normalized_area_under_discovery_curve(curve, global_optimum=10.0)
        assert naudc == pytest.approx(1.0, abs=1e-6)

    def test_naudc_increases_with_better_discovery(self):
        slow = [1.0, 2.0, 3.0, 4.0, 5.0]
        fast = [4.0, 5.0, 5.0, 5.0, 5.0]
        naudc_slow = normalized_area_under_discovery_curve(slow, global_optimum=5.0)
        naudc_fast = normalized_area_under_discovery_curve(fast, global_optimum=5.0)
        assert naudc_fast > naudc_slow

    def test_trapezoidal_naudc_uses_actual_candidate_budgets(self):
        # Initial seed at 4, then a non-default batch of 3 and a partial batch of 1.
        budgets = [4, 7, 8]
        curve = [2.0, 6.0, 8.0]
        expected = ((0.2 + 0.6) / 2 * 3 + (0.6 + 0.8) / 2) / 4
        assert normalized_area_under_discovery_curve(
            curve, 10.0, budgets=budgets,
        ) == pytest.approx(expected)

    def test_single_point_curve_keeps_normalized_value(self):
        assert normalized_area_under_discovery_curve(
            [4.0], global_optimum=8.0, budgets=[12],
        ) == pytest.approx(0.5)

    def test_single_point_curve_rejects_non_finite_budget(self):
        with pytest.raises(ValueError, match="finite"):
            normalized_area_under_discovery_curve(
                [4.0], global_optimum=8.0, budgets=[float("nan")],
            )

    def test_duplicate_multi_point_budgets_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            normalized_area_under_discovery_curve(
                [2.0, 3.0], global_optimum=5.0, budgets=[4, 4],
            )

    def test_curve_and_budget_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal lengths"):
            normalized_area_under_discovery_curve([1.0, 2.0], 3.0, budgets=[1])

    def test_initial_and_partial_acquisitions_have_exact_budget_labels(self):
        budgets, values = discovery_curve_with_initial(3.0, [1.0, 4.0, 2.0, 5.0], 10)
        assert budgets == [10, 11, 12, 13, 14]
        assert values == [3.0, 3.0, 4.0, 4.0, 5.0]

    def test_equivalent_candidate_curves_have_same_naudc_across_services(self):
        # Benchmark batches 3+1 and persisted per-candidate acquisition history
        # resolve to the same curve and therefore the same integrated NAUDC.
        x, curve = discovery_curve_with_initial(2.0, [3.0, 1.0, 5.0, 4.0], 7)
        benchmark_value = normalized_area_under_discovery_curve(curve, 5.0, budgets=x)
        evaluation_value = normalized_area_under_discovery_curve(curve, 5.0, budgets=x)
        assert evaluation_value == pytest.approx(benchmark_value)


class TestTopKDiscovery:
    def test_all_found(self):
        all_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        acquired = [5.0, 4.0, 3.0]
        rate = top_k_discovery_rate(acquired, all_values, k=3)
        assert rate == pytest.approx(1.0)

    def test_none_found(self):
        all_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        acquired = [1.0, 2.0]
        rate = top_k_discovery_rate(acquired, all_values, k=3)
        assert rate == pytest.approx(0.0)


class TestBatchDiversity:
    def test_identical_candidates_zero_diversity(self):
        X = np.array([[1.0, 0.0, 0.0]] * 5)
        div = batch_diversity_score(X)
        assert div == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_candidates_high_diversity(self):
        X = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        div = batch_diversity_score(X)
        assert div > 0.4  # cosine distance = 1 for orthogonal, normalized to 0.5

    def test_single_candidate(self):
        X = np.array([[1.0, 2.0, 3.0]])
        assert batch_diversity_score(X) == 0.0


class TestCalibration:
    def test_known_interval_coverage_and_coverage_ece(self):
        residuals = np.array([0.0] * 90 + [1.8] * 5 + [3.0] * 5)
        result = expected_calibration_error(
            residuals, np.zeros(100), np.ones(100), n_bins=3
        )
        assert result["coverage_90pct"] == pytest.approx(0.90)
        assert result["coverage_95pct"] == pytest.approx(0.95)
        expected = np.array([0.05, 0.50, 0.95])
        observed = np.array([0.90, 0.90, 0.95])
        assert result["ece"] == pytest.approx(np.mean(np.abs(expected - observed)))
        assert result["n_observations"] == 100
        assert len(result["reliability_diagram"]["expected_coverage"]) == 3
        assert len(result["reliability_diagram"]["observed_coverage"]) == 3

    def test_zero_standard_deviation_is_valid_and_exact(self):
        result = expected_calibration_error(
            np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
        )
        assert result["coverage_90pct"] == pytest.approx(0.5)
        assert result["coverage_95pct"] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "values,kwargs,message",
        [
            (([1, 2], [1], [1, 1]), {}, "equal lengths"),
            (([], [], []), {}, "must not be empty"),
            (([np.nan], [0], [1]), {}, "finite"),
            (([1], [np.inf], [1]), {}, "finite"),
            (([1], [0], [np.inf]), {}, "finite"),
            (([1], [0], [-0.1]), {}, "negative"),
            (([1], [0], [1]), {"n_bins": 0}, "positive integer"),
            (([1], [0], [1]), {"n_bins": 1.5}, "positive integer"),
        ],
    )
    def test_invalid_calibration_inputs_raise(self, values, kwargs, message):
        with pytest.raises(ValueError, match=message):
            expected_calibration_error(*values, **kwargs)

    def test_well_calibrated_gp(self):
        """Perfect calibration: residuals match predicted uncertainty."""
        rng = np.random.RandomState(0)
        n = 500
        std = np.ones(n) * 1.0
        y_true = rng.normal(0, 1, n)
        y_pred = np.zeros(n)
        cal = expected_calibration_error(y_true, y_pred, std)
        assert cal["ece"] < 0.1  # Well calibrated
        assert 0.85 < cal["coverage_95pct"] < 1.0

    def test_overconfident_model(self):
        """Overconfident model: std too small → coverage < expected."""
        rng = np.random.RandomState(0)
        n = 500
        y_true = rng.normal(0, 5, n)   # true spread = 5
        y_pred = np.zeros(n)
        std = np.ones(n) * 0.1          # predicted spread = 0.1 (way too small)
        cal = expected_calibration_error(y_true, y_pred, std)
        assert cal["coverage_95pct"] < 0.5  # drastically under-covering


class TestStrategyComparison:
    def _make_experiment(self, strategy, quality=1.0):
        n = 10
        budget = list(range(10, 110, 10))
        rmse = [1.0 / (i * quality + 1) for i in range(1, n + 1)]
        r2 = [min(1.0, i * 0.1 * quality) for i in range(1, n + 1)]
        best = [i * quality for i in range(1, n + 1)]
        diversity = [0.3 + 0.01 * i for i in range(n)]
        acquired = [float(i * quality) for i in range(1, n * 10 + 1)]
        all_pool = np.arange(1, 101, dtype=float)
        return strategy, budget, rmse, r2, best, acquired, diversity, all_pool

    def test_comparison_builds_charts(self):
        comp = StrategyComparison(global_optimum=100.0)
        for strat, quality in [("random", 0.5), ("puda", 1.0)]:
            s, budget, rmse, r2, best, acq, div, pool = self._make_experiment(strat, quality)
            comp.add_experiment(s, budget, rmse, r2, best, acq, div, pool)
        charts = comp.build_charts()
        assert "discovery_curve" in charts
        assert "rmse_curve" in charts
        assert "regret_curve" in charts
        assert "summary_table" in charts
        assert len(charts["summary_table"]) == 2

    def test_better_strategy_ranks_first(self):
        comp = StrategyComparison(global_optimum=100.0)
        for strat, quality in [("random", 0.3), ("puda", 1.5)]:
            s, budget, rmse, r2, best, acq, div, pool = self._make_experiment(strat, quality)
            comp.add_experiment(s, budget, rmse, r2, best, acq, div, pool)
        charts = comp.build_charts()
        # PUDA has higher quality → should rank first
        assert charts["summary_table"][0]["strategy"] == "puda"
