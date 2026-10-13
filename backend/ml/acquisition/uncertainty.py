"""
Strategy 2 — Uncertainty Sampling.

Selects candidates with the highest predictive uncertainty σ(x).

For a GP:   σ(x) = posterior standard deviation (exact)
For RF:     σ(x) = std of individual tree predictions (proxy)

Pure exploration: completely ignores predicted performance.

This strategy is optimal when:
- The model is severely under-informed (very few labeled samples)
- The goal is to reduce uncertainty uniformly across the feature space
- Exploration is more valuable than exploitation

It fails when:
- Many high-uncertainty candidates have poor predicted performance
- The objective is to find the BEST material, not the most uncertain one

Used as Baseline 2 (after random) to isolate the effect of uncertainty
information from Bayesian acquisition function design.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class UncertaintyAcquisition:
    """
    Uncertainty sampling: select highest σ(x) candidates.

    Parameters
    ----------
    diversity_penalty : float
        If > 0, apply a cosine-distance diversity penalty within the batch
        to avoid selecting near-identical high-uncertainty candidates.
        Range: [0.0, 1.0]. 0 = pure uncertainty, 1 = pure diversity.
    """

    name = "uncertainty"
    label = "Uncertainty Sampling"

    def __init__(self, diversity_penalty: float = 0.0, random_state: int = 42):
        self.diversity_penalty = diversity_penalty
        self.random_state = random_state

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        X_selected: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute acquisition scores = normalized σ(x).

        If diversity_penalty > 0 and X_selected is provided, penalize
        candidates that are too similar to already-selected ones.
        """
        scores = _normalize(std)

        if self.diversity_penalty > 0.0 and X_selected is not None and len(X_selected) > 0:
            div_scores = _min_cosine_distance(X_pool, X_selected)
            scores = (1 - self.diversity_penalty) * scores + self.diversity_penalty * div_scores

        return scores

    def select_batch(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        batch_size: int,
        already_selected: list | None = None,
    ) -> np.ndarray:
        """
        Greedily select `batch_size` highest-uncertainty candidates.
        Updates diversity scores after each pick if diversity_penalty > 0.
        """
        excluded = set(already_selected or [])
        available = [i for i in range(len(mean)) if i not in excluded]
        n_select = min(batch_size, len(available))

        selected = []
        X_selected_so_far = np.empty((0, X_pool.shape[1]))

        for _ in range(n_select):
            avail_arr = np.array([i for i in available if i not in selected])
            scores = self.score(
                mean[avail_arr],
                std[avail_arr],
                X_pool[avail_arr],
                X_selected_so_far if len(X_selected_so_far) > 0 else None,
            )
            best_local = int(np.argmax(scores))
            best_global = int(avail_arr[best_local])
            selected.append(best_global)
            X_selected_so_far = np.vstack([X_selected_so_far, X_pool[best_global]])

        logger.info(
            "[Uncertainty] Selected %d candidates | "
            "mean σ of selected: %.4f",
            len(selected),
            float(std[selected].mean()) if selected else 0.0,
        )
        return np.array(selected)

    def breakdown(self, idx: int, mean: np.ndarray, std: np.ndarray,
                  X_pool: np.ndarray) -> dict:
        uncertainty_normalized = float(_normalize(std)[idx])
        return {
            "strategy": self.label,
            "predicted_performance": float(mean[idx]),
            "uncertainty": float(std[idx]),
            "uncertainty_normalized": uncertainty_normalized,
            "exploration_contribution": uncertainty_normalized,
            "diversity_contribution": 0.0,
            "final_score": uncertainty_normalized,
            "selection_reason": (
                f"Selected because its predictive uncertainty σ={std[idx]:.4f} eV "
                f"is among the highest in the pool. The model has seen few similar "
                f"materials, so evaluating this candidate provides maximum new information."
            ),
        }


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns 0.5 if all values are identical."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - mn) / (mx - mn)


def _min_cosine_distance(X_pool: np.ndarray, X_selected: np.ndarray) -> np.ndarray:
    """
    For each row in X_pool, compute minimum cosine distance to X_selected.
    Returns values in [0, 1]; higher = more diverse.
    """
    from sklearn.metrics.pairwise import cosine_distances
    dists = cosine_distances(X_pool, X_selected)
    min_dists = dists.min(axis=1)
    return _normalize(min_dists)
