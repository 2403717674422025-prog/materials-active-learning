"""
Strategy 1 — Random Sampling (baseline).

Selects candidates uniformly at random from the unlabeled pool.
No model predictions are used.

This is the hardest baseline to beat: it represents what happens when
the experimental budget is spent without any intelligent guidance.
Any active learning strategy that does NOT outperform random sampling
provides no value over blind exploration.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RandomAcquisition:
    """
    Random acquisition strategy.

    Parameters
    ----------
    random_state : int
        Seed for the random number generator. Fixed per experiment for
        reproducibility.
    """

    name = "random"
    label = "Random Sampling"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        X_selected: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return random scores uniformly drawn from [0, 1].

        Parameters
        ----------
        mean : np.ndarray, shape (n_pool,)
            Predicted means (not used, kept for API consistency).
        std : np.ndarray, shape (n_pool,)
            Predicted stds (not used, kept for API consistency).
        X_pool : np.ndarray, shape (n_pool, n_features)
            Feature matrix of unlabeled candidates.
        X_selected : np.ndarray, optional
            Already-selected candidates in the current batch (not used).

        Returns
        -------
        scores : np.ndarray, shape (n_pool,)
            Random scores in [0, 1].
        """
        return self._rng.uniform(0.0, 1.0, size=len(mean))

    def select_batch(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        batch_size: int,
        already_selected: list | None = None,
    ) -> np.ndarray:
        """
        Select `batch_size` candidate indices by random sampling.

        Parameters
        ----------
        already_selected : list
            Pool indices already selected in earlier iterations (excluded).

        Returns
        -------
        selected_indices : np.ndarray, shape (batch_size,)
            Indices into X_pool of the selected candidates.
        """
        excluded = set(already_selected or [])
        available = [i for i in range(len(mean)) if i not in excluded]
        n_select = min(batch_size, len(available))
        chosen = self._rng.choice(available, size=n_select, replace=False)
        logger.info(
            "[Random] Selected %d candidates from pool of %d",
            n_select, len(available),
        )
        return chosen

    def breakdown(self, idx: int, mean: np.ndarray, std: np.ndarray,
                  X_pool: np.ndarray) -> dict:
        """Per-candidate acquisition breakdown for the UI explainer page."""
        return {
            "strategy": self.label,
            "predicted_performance": float(mean[idx]),
            "uncertainty": float(std[idx]),
            "exploration_contribution": 0.0,
            "diversity_contribution": 0.0,
            "final_score": float(self._rng.uniform(0, 1)),
            "selection_reason": "Selected randomly. No model information used.",
        }
