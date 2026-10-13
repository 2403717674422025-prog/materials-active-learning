"""
Strategy 4 — Proposed Hybrid Strategy: PUDA
(Prediction + Uncertainty + Diversity Acquisition)

Motivation
----------
Standard acquisition functions have blind spots:
  - EI/UCB: ignores diversity → batch contains near-identical candidates
  - Uncertainty: ignores predicted performance → wastes budget on bad materials
  - Random: ignores all information

PUDA combines three complementary signals:

    PUDA(x) = α · norm(μ(x))
            + β · norm(σ(x))
            + γ · norm(div(x, S_selected))

Where:
    α = weight on predicted performance  (exploitation)
    β = weight on predictive uncertainty (exploration)
    γ = weight on diversity              (coverage)
    α + β + γ = 1

Each component is min-max normalized over the current pool so the
weights have consistent meaning across iterations.

Diversity score
---------------
    div(x, S) = min_{s ∈ S} cosine_distance(x, s)

This is computed greedily: after each candidate is added to S,
the diversity scores for all remaining candidates are recomputed.
This implements a greedy k-center approximation that produces
a batch covering diverse regions of feature space.

Greedy batch selection algorithm
----------------------------------
    S = {}
    for i = 1 to batch_size:
        for each x in pool:
            div(x) = min cosine_distance(x, s) for s in S
                     (= 1.0 if S is empty → no penalty for first pick)
        PUDA(x) = α·norm(μ) + β·norm(σ) + γ·norm(div)
        x* = argmax PUDA(x) over pool
        S  = S ∪ {x*}
        pool = pool \ {x*}

This runs in O(batch_size · n_pool · |features|) — efficient for batch
sizes up to ~100 and pool sizes up to ~10,000.

Weight interpretation
---------------------
    α=0.4, β=0.35, γ=0.25  (defaults)

    A slight emphasis on predicted performance reflects the goal of finding
    high-performing materials. Uncertainty and diversity weights are set
    nearly equal to encourage balanced exploration coverage.

    These defaults are validated by the ablation study in experiments/.
    Users can adjust them via the experiment setup UI.

Scientific claim
----------------
PUDA is expected to outperform pure uncertainty sampling and random
sampling in discovery efficiency. Its advantage over EI is NOT
guaranteed and is tested experimentally in Experiment C.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns 0.5 if range is zero."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-10:
        return np.full(len(arr), 0.5, dtype=float)
    return (arr - mn) / (mx - mn)


def _diversity_scores(
    X_candidates: np.ndarray,
    X_selected: np.ndarray,
) -> np.ndarray:
    """
    Compute minimum cosine distance from each candidate to the selected set.

    Returns 1.0 for all candidates if X_selected is empty.
    Values in [0, 1] (cosine distance range).
    """
    if len(X_selected) == 0:
        return np.ones(len(X_candidates), dtype=float)
    dists = cosine_distances(X_candidates, X_selected)
    return dists.min(axis=1)


class PUDAAcquisition:
    """
    PUDA: Prediction + Uncertainty + Diversity Acquisition.

    Parameters
    ----------
    alpha : float
        Weight for normalized predicted performance.
    beta : float
        Weight for normalized predictive uncertainty.
    gamma : float
        Weight for normalized diversity score.
    mode : str
        'maximize' or 'minimize' — direction of the property target.
    normalize_weights : bool
        If True, re-normalize α, β, γ so they sum to 1.
    random_state : int
        Seed for any tie-breaking randomness.
    """

    name = "puda"
    label = "PUDA Hybrid (Proposed)"

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.35,
        gamma: float = 0.25,
        mode: str = "maximize",
        normalize_weights: bool = True,
        random_state: int = 42,
    ):
        if normalize_weights:
            total = alpha + beta + gamma
            if total < 1e-9:
                raise ValueError("PUDA weights must not all be zero.")
            alpha, beta, gamma = alpha / total, beta / total, gamma / total

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mode = mode
        self.random_state = random_state
        self._last_scores: dict = {}

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        X_selected: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute PUDA scores for candidates in X_pool.

        Parameters
        ----------
        mean : np.ndarray  shape (n,)  predicted means
        std  : np.ndarray  shape (n,)  predicted stds
        X_pool : np.ndarray  shape (n, d)  feature matrix
        X_selected : np.ndarray  shape (m, d)  already-selected this batch

        Returns
        -------
        scores : np.ndarray  shape (n,)  PUDA acquisition values in [0, 1]
        """
        n = len(mean)
        X_sel = X_selected if X_selected is not None else np.empty((0, X_pool.shape[1]))

        # Component 1: predicted performance
        if self.mode == "maximize":
            perf_raw = mean.copy()
        else:
            perf_raw = -mean.copy()
        perf_norm = _normalize(perf_raw)

        # Component 2: predictive uncertainty
        unc_norm = _normalize(std)

        # Component 3: diversity
        div_raw = _diversity_scores(X_pool, X_sel)
        div_norm = _normalize(div_raw)

        scores = self.alpha * perf_norm + self.beta * unc_norm + self.gamma * div_norm

        # Cache for breakdown later
        self._last_scores = {
            "perf_norm": perf_norm,
            "unc_norm": unc_norm,
            "div_norm": div_norm,
            "scores": scores,
        }

        return scores

    def select_batch(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        batch_size: int,
        already_selected: list | None = None,
        y_labeled: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Greedy batch selection via iterative PUDA scoring.

        After each pick, the selected candidate is added to X_selected_so_far
        and diversity scores are recomputed for remaining candidates.

        Returns
        -------
        selected_global_indices : np.ndarray  shape (batch_size,)
            Indices into the original pool arrays.
        """
        excluded = set(already_selected or [])
        available = np.array([i for i in range(len(mean)) if i not in excluded])
        n_select = min(batch_size, len(available))

        selected_global: List[int] = []
        X_selected_so_far = np.empty((0, X_pool.shape[1]))

        # Per-candidate breakdown storage for UI
        self._breakdown_store: dict = {}

        for step in range(n_select):
            # Remaining available indices (exclude already picked in this batch)
            remaining = np.array([i for i in available if i not in selected_global])
            if len(remaining) == 0:
                break

            scores = self.score(
                mean=mean[remaining],
                std=std[remaining],
                X_pool=X_pool[remaining],
                X_selected=X_selected_so_far if len(X_selected_so_far) > 0 else None,
            )

            best_local = int(np.argmax(scores))
            best_global = int(remaining[best_local])

            # Store breakdown for this candidate
            self._breakdown_store[best_global] = {
                "perf_norm": float(self._last_scores["perf_norm"][best_local]),
                "unc_norm": float(self._last_scores["unc_norm"][best_local]),
                "div_norm": float(self._last_scores["div_norm"][best_local]),
                "final_score": float(scores[best_local]),
                "selection_rank": step + 1,
            }

            selected_global.append(best_global)
            X_selected_so_far = np.vstack([X_selected_so_far, X_pool[best_global]])

        logger.info(
            "[PUDA] α=%.2f β=%.2f γ=%.2f | Selected %d candidates | "
            "mean score=%.4f",
            self.alpha, self.beta, self.gamma,
            len(selected_global),
            float(np.mean([self._breakdown_store[i]["final_score"]
                           for i in selected_global])) if selected_global else 0.0,
        )
        return np.array(selected_global, dtype=int)

    def breakdown(
        self,
        idx: int,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        X_selected: np.ndarray | None = None,
    ) -> dict:
        """
        Return per-candidate acquisition breakdown for the UI explainer.

        This makes the PUDA selection decision fully transparent.
        """
        # Compute fresh scores for this candidate
        self.score(
            mean=mean,
            std=std,
            X_pool=X_pool,
            X_selected=X_selected,
        )

        perf_n = float(self._last_scores["perf_norm"][idx])
        unc_n = float(self._last_scores["unc_norm"][idx])
        div_n = float(self._last_scores["div_norm"][idx])
        final = float(self._last_scores["scores"][idx])

        contrib_perf = self.alpha * perf_n
        contrib_unc = self.beta * unc_n
        contrib_div = self.gamma * div_n

        return {
            "strategy": self.label,
            "predicted_performance": float(mean[idx]),
            "prediction_normalized": perf_n,
            "uncertainty": float(std[idx]),
            "uncertainty_normalized": unc_n,
            "diversity_normalized": div_n,
            "exploration_contribution": float(contrib_unc),
            "exploitation_contribution": float(contrib_perf),
            "diversity_contribution": float(contrib_div),
            "final_score": final,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "formula": (
                f"PUDA = {self.alpha:.2f}×{perf_n:.3f} "
                f"+ {self.beta:.2f}×{unc_n:.3f} "
                f"+ {self.gamma:.2f}×{div_n:.3f} "
                f"= {final:.4f}"
            ),
            "selection_reason": (
                f"PUDA score {final:.4f}: "
                f"Predicted performance contributes {contrib_perf:.3f} (α×norm_μ), "
                f"uncertainty contributes {contrib_unc:.3f} (β×norm_σ), "
                f"diversity contributes {contrib_div:.3f} (γ×cosine_dist). "
                f"This candidate balances high predicted band gap, "
                f"informative uncertainty, and feature-space diversity."
            ),
        }


class AblationPUDA(PUDAAcquisition):
    """
    PUDA variant for ablation study.

    Allows disabling individual components to measure their contribution:
        - Prediction only:   α=1, β=0, γ=0
        - Pred + Unc:        α=0.5, β=0.5, γ=0
        - Pred + Div:        α=0.5, β=0, γ=0.5
        - Full (Pred+Unc+Div): α=0.4, β=0.35, γ=0.25  (default)
    """

    ABLATION_CONFIGS = {
        "pred_only": (1.0, 0.0, 0.0),
        "pred_unc": (0.5, 0.5, 0.0),
        "pred_div": (0.5, 0.0, 0.5),
        "full": (0.4, 0.35, 0.25),
    }

    def __init__(self, ablation_mode: str = "full", **kwargs):
        if ablation_mode not in self.ABLATION_CONFIGS:
            raise ValueError(
                f"Unknown ablation mode '{ablation_mode}'. "
                f"Available: {list(self.ABLATION_CONFIGS.keys())}"
            )
        alpha, beta, gamma = self.ABLATION_CONFIGS[ablation_mode]
        self.ablation_mode = ablation_mode
        super().__init__(alpha=alpha, beta=beta, gamma=gamma,
                         normalize_weights=False, **kwargs)

    @property
    def label(self) -> str:  # type: ignore[override]
        return f"PUDA ({self.ablation_mode})"
