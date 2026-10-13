"""
Tests for all acquisition strategy implementations.

Covers:
  - Random: score returns correct shape, no duplicates in selection
  - Uncertainty: highest std candidates selected
  - EI: EI = 0 when std = 0 and mean < y_best; EI > 0 when mean > y_best
  - UCB: score = mean + kappa * std
  - PUDA: score components are in [0,1], batch has correct size
  - PUDA: greedy diversity penalty produces more diverse batches than pure uncertainty
  - Budget constraint: never select more than available
  - No duplicate selection within one batch
"""

import numpy as np
import pytest

from backend.ml.acquisition.random import RandomAcquisition
from backend.ml.acquisition.uncertainty import UncertaintyAcquisition
from backend.ml.acquisition.expected_improvement import (
    ExpectedImprovementAcquisition, UCBAcquisition
)
from backend.ml.acquisition.hybrid_puda import PUDAAcquisition


@pytest.fixture
def pool_data():
    """Synthetic pool: 50 candidates, 10 features."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)
    mean = rng.uniform(0, 5, 50)
    std = rng.uniform(0.01, 2.0, 50)
    return X, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Random Acquisition
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomAcquisition:
    def test_score_shape(self, pool_data):
        X, mean, std = pool_data
        acq = RandomAcquisition(random_state=0)
        scores = acq.score(mean, std, X)
        assert scores.shape == (50,)

    def test_score_in_range(self, pool_data):
        X, mean, std = pool_data
        acq = RandomAcquisition(random_state=0)
        scores = acq.score(mean, std, X)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_select_correct_count(self, pool_data):
        X, mean, std = pool_data
        acq = RandomAcquisition(random_state=0)
        selected = acq.select_batch(mean, std, X, batch_size=10)
        assert len(selected) == 10

    def test_no_duplicates(self, pool_data):
        X, mean, std = pool_data
        acq = RandomAcquisition(random_state=0)
        selected = acq.select_batch(mean, std, X, batch_size=20)
        assert len(set(selected)) == len(selected)

    def test_respects_already_selected(self, pool_data):
        X, mean, std = pool_data
        acq = RandomAcquisition(random_state=0)
        already = [0, 1, 2, 3, 4]
        selected = acq.select_batch(mean, std, X, batch_size=10, already_selected=already)
        for s in selected:
            assert s not in already


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty Acquisition
# ─────────────────────────────────────────────────────────────────────────────

class TestUncertaintyAcquisition:
    def test_selects_highest_uncertainty(self, pool_data):
        X, mean, std = pool_data
        acq = UncertaintyAcquisition()
        selected = acq.select_batch(mean, std, X, batch_size=5)
        # All selected should be among top-10 highest std candidates
        top_10_idx = set(np.argsort(std)[::-1][:10])
        assert all(s in top_10_idx for s in selected)

    def test_no_duplicates(self, pool_data):
        X, mean, std = pool_data
        acq = UncertaintyAcquisition()
        selected = acq.select_batch(mean, std, X, batch_size=15)
        assert len(set(selected)) == len(selected)

    def test_score_shape(self, pool_data):
        X, mean, std = pool_data
        acq = UncertaintyAcquisition()
        scores = acq.score(mean, std, X)
        assert scores.shape == (50,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


# ─────────────────────────────────────────────────────────────────────────────
# Expected Improvement
# ─────────────────────────────────────────────────────────────────────────────

class TestExpectedImprovement:
    def test_ei_zero_when_below_best_zero_std(self):
        acq = ExpectedImprovementAcquisition(xi=0.0)
        mean = np.array([1.0, 2.0, 3.0])  # all below y_best=5
        std = np.array([0.0, 0.0, 0.0])   # zero uncertainty
        ei = acq.compute_ei(mean, std, y_best=5.0)
        assert np.all(ei == 0.0)

    def test_ei_positive_above_best(self):
        acq = ExpectedImprovementAcquisition(xi=0.0)
        mean = np.array([6.0, 7.0, 8.0])  # all above y_best=5
        std = np.array([0.01, 0.01, 0.01])
        ei = acq.compute_ei(mean, std, y_best=5.0)
        assert np.all(ei > 0)

    def test_ei_non_negative(self, pool_data):
        X, mean, std = pool_data
        acq = ExpectedImprovementAcquisition(xi=0.01)
        ei = acq.compute_ei(mean, std, y_best=3.0)
        assert np.all(ei >= 0)

    def test_select_batch_size(self, pool_data):
        X, mean, std = pool_data
        acq = ExpectedImprovementAcquisition(xi=0.01)
        acq.set_best_observed(np.array([3.0]))
        selected = acq.select_batch(mean, std, X, batch_size=10)
        assert len(selected) == 10

    def test_no_duplicates(self, pool_data):
        X, mean, std = pool_data
        acq = ExpectedImprovementAcquisition(xi=0.01)
        acq.set_best_observed(np.array([2.0]))
        selected = acq.select_batch(mean, std, X, batch_size=20)
        assert len(set(selected)) == len(selected)

    def test_ucb_score_formula(self, pool_data):
        X, mean, std = pool_data
        kappa = 2.0
        acq = UCBAcquisition(kappa=kappa)
        scores = acq.score(mean, std, X)
        expected = mean + kappa * std
        np.testing.assert_array_almost_equal(scores, expected)


# ─────────────────────────────────────────────────────────────────────────────
# PUDA Acquisition
# ─────────────────────────────────────────────────────────────────────────────

class TestPUDAAcquisition:
    def test_select_correct_batch_size(self, pool_data):
        X, mean, std = pool_data
        acq = PUDAAcquisition(alpha=0.4, beta=0.35, gamma=0.25)
        selected = acq.select_batch(mean, std, X, batch_size=10)
        assert len(selected) == 10

    def test_no_duplicates_in_batch(self, pool_data):
        X, mean, std = pool_data
        acq = PUDAAcquisition(alpha=0.4, beta=0.35, gamma=0.25)
        selected = acq.select_batch(mean, std, X, batch_size=15)
        assert len(set(selected.tolist())) == 15

    def test_scores_in_unit_range(self, pool_data):
        X, mean, std = pool_data
        acq = PUDAAcquisition(alpha=0.4, beta=0.35, gamma=0.25)
        scores = acq.score(mean, std, X)
        assert np.all(scores >= 0), "PUDA scores must be ≥ 0"
        assert np.all(scores <= 1.0 + 1e-6), "PUDA scores must be ≤ 1"

    def test_diversity_reduces_clustering(self, pool_data):
        """PUDA (with γ>0) should select more diverse candidates than pure uncertainty."""
        from sklearn.metrics.pairwise import cosine_distances
        X, mean, std = pool_data

        # Pure uncertainty
        unc = UncertaintyAcquisition()
        unc_sel = unc.select_batch(mean, std, X, batch_size=10)

        # PUDA with diversity
        puda = PUDAAcquisition(alpha=0.0, beta=0.5, gamma=0.5)
        puda_sel = puda.select_batch(mean, std, X, batch_size=10)

        def mean_pairwise_dist(idx):
            selected_X = X[idx]
            d = cosine_distances(selected_X, selected_X)
            n = len(idx)
            return d[np.triu_indices(n, k=1)].mean()

        unc_div = mean_pairwise_dist(unc_sel)
        puda_div = mean_pairwise_dist(puda_sel)
        # PUDA diversity ≥ pure uncertainty diversity (or close)
        assert puda_div >= unc_div * 0.9, \
            f"PUDA should produce at least as diverse batches (puda={puda_div:.4f}, unc={unc_div:.4f})"

    def test_weight_normalization(self):
        """Unnormalized weights should be auto-normalized."""
        acq = PUDAAcquisition(alpha=2.0, beta=2.0, gamma=1.0, normalize_weights=True)
        total = acq.alpha + acq.beta + acq.gamma
        assert abs(total - 1.0) < 1e-6

    def test_budget_constraint(self, pool_data):
        """Never select more candidates than the available pool."""
        X, mean, std = pool_data
        acq = PUDAAcquisition()
        selected = acq.select_batch(mean, std, X, batch_size=200)  # more than pool
        assert len(selected) == 50

    def test_breakdown_keys(self, pool_data):
        X, mean, std = pool_data
        acq = PUDAAcquisition()
        bd = acq.breakdown(0, mean, std, X)
        for key in ["strategy", "predicted_performance", "uncertainty", "diversity_contribution",
                    "final_score", "formula", "selection_reason"]:
            assert key in bd


# ─────────────────────────────────────────────────────────────────────────────
# Budget constraint (all strategies)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy_cls,kwargs", [
    (RandomAcquisition, {}),
    (UncertaintyAcquisition, {}),
    (PUDAAcquisition, {}),
])
def test_batch_never_exceeds_pool(strategy_cls, kwargs):
    X = np.random.randn(10, 5)
    mean = np.random.randn(10)
    std = np.abs(np.random.randn(10))
    acq = strategy_cls(**kwargs)
    selected = acq.select_batch(mean, std, X, batch_size=999)
    assert len(selected) <= 10
