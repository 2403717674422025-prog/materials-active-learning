"""
Strategy 3 — Bayesian Optimization via Expected Improvement (EI).

Mathematical derivation
-----------------------
Let f(x) be the unknown property function modeled by a surrogate GP.
Let y* = max(y_labeled) be the best observed value so far.
We seek to maximize f, so EI is defined as:

    EI(x) = E[max(f(x) - y*, 0)]

For a Gaussian surrogate with predictive distribution N(μ(x), σ²(x)):

    EI(x) = (μ(x) - y* - ξ) · Φ(Z) + σ(x) · φ(Z)

    where:
        Z     = (μ(x) - y* - ξ) / σ(x)
        Φ(·)  = standard normal CDF
        φ(·)  = standard normal PDF
        ξ ≥ 0 = exploration parameter (jitter)

Exploration-exploitation tradeoff
----------------------------------
    ξ = 0.0 : pure exploitation (select highest μ(x) > y*)
    ξ > 0   : shifts the threshold down, giving more weight to
              high-uncertainty candidates even when μ(x) < y*

Key insight: EI naturally balances exploration and exploitation —
  - When σ(x) → 0: EI → max(μ(x) - y*, 0)  (pure exploitation)
  - When σ(x) >> 0: the σ·φ(Z) term dominates  (exploration bonus)

UCB (Upper Confidence Bound) alternative
-----------------------------------------
    UCB(x) = μ(x) + κ · σ(x)

    κ controls the exploration-exploitation balance directly.
    UCB is simpler, easier to interpret, and theoretically motivated
    (sub-linear regret under certain kernel conditions).
    Both are available here; EI is the default.

Reference
---------
Jones, D.R., Schonlau, M., Welch, W.J. (1998). Efficient global
optimization of expensive black-box functions. J. Global Optim. 13:455-492.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class ExpectedImprovementAcquisition:
    """
    Expected Improvement acquisition function.

    Parameters
    ----------
    xi : float
        Exploration parameter ξ ≥ 0. Default 0.01 provides a small
        exploration bonus without sacrificing exploitation.
    mode : str
        'maximize' or 'minimize'. For band gap discovery we maximize.
    """

    name = "ei"
    label = "Expected Improvement (Bayesian)"

    def __init__(self, xi: float = 0.01, mode: str = "maximize", random_state: int = 42):
        self.xi = xi
        self.mode = mode
        self.random_state = random_state
        self._best_so_far: float | None = None

    def set_best_observed(self, y_labeled: np.ndarray) -> None:
        """Update y* from the current labeled set."""
        if self.mode == "maximize":
            self._best_so_far = float(np.max(y_labeled))
        else:
            self._best_so_far = float(np.min(y_labeled))

    def compute_ei(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_best: float,
    ) -> np.ndarray:
        """
        Compute EI scores for all pool candidates.

        Parameters
        ----------
        mean : np.ndarray  shape (n,)
        std  : np.ndarray  shape (n,)
        y_best : float     best observed value so far

        Returns
        -------
        ei : np.ndarray, shape (n,)  EI scores ≥ 0
        """
        std_safe = np.maximum(std, 1e-9)  # prevent division by zero

        if self.mode == "maximize":
            improvement = mean - y_best - self.xi
        else:
            improvement = y_best - mean - self.xi

        Z = improvement / std_safe
        ei = improvement * norm.cdf(Z) + std_safe * norm.pdf(Z)

        # Candidates where std ≈ 0 and improvement < 0 get EI = 0
        ei = np.where(std < 1e-9, np.maximum(improvement, 0.0), ei)
        ei = np.maximum(ei, 0.0)
        return ei

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        X_selected: np.ndarray | None = None,
        y_labeled: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute acquisition scores (EI values).

        Parameters
        ----------
        y_labeled : np.ndarray, optional
            Current labeled set values. If None, uses pre-set _best_so_far.
        """
        if y_labeled is not None:
            self.set_best_observed(y_labeled)

        if self._best_so_far is None:
            logger.warning("EI: best_so_far not set. Falling back to uncertainty.")
            return _normalize(std)

        ei = self.compute_ei(mean, std, self._best_so_far)
        return ei

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
        Greedy batch selection by EI score.

        Note: Standard EI is designed for sequential (one-at-a-time) selection.
        In the batch setting we use a greedy approximation: select the top-k
        by EI, without re-computing after each pick.
        """
        excluded = set(already_selected or [])
        available = np.array([i for i in range(len(mean)) if i not in excluded])
        n_select = min(batch_size, len(available))

        if y_labeled is not None:
            self.set_best_observed(y_labeled)

        scores = self.compute_ei(mean[available], std[available], self._best_so_far)
        top_local = np.argsort(scores)[::-1][:n_select]
        selected = available[top_local]

        logger.info(
            "[EI] y*=%.4f | Selected %d candidates | "
            "mean EI of selected: %.6f",
            self._best_so_far or 0.0,
            len(selected),
            float(scores[top_local].mean()) if len(scores) > 0 else 0.0,
        )
        return selected

    def breakdown(self, idx: int, mean: np.ndarray, std: np.ndarray,
                  X_pool: np.ndarray, y_best: float | None = None) -> dict:
        y_best = y_best or self._best_so_far or 0.0
        ei_val = float(self.compute_ei(mean[idx:idx+1], std[idx:idx+1], y_best)[0])
        improvement = float(mean[idx]) - y_best - self.xi
        Z = improvement / max(float(std[idx]), 1e-9)
        exploit = improvement * float(norm.cdf(Z))
        explore = float(std[idx]) * float(norm.pdf(Z))

        return {
            "strategy": self.label,
            "predicted_performance": float(mean[idx]),
            "uncertainty": float(std[idx]),
            "exploration_contribution": float(max(explore, 0)),
            "diversity_contribution": 0.0,
            "exploitation_contribution": float(max(exploit, 0)),
            "final_score": ei_val,
            "y_best": float(y_best),
            "xi": self.xi,
            "Z": float(Z),
            "selection_reason": (
                f"EI={ei_val:.6f} eV. Predicted mean {mean[idx]:.4f} eV vs current best "
                f"{y_best:.4f} eV. Exploit component: {max(exploit,0):.4f}, "
                f"Explore component: {max(explore,0):.4f} (σ={std[idx]:.4f} eV). "
                f"EI balances the predicted improvement over y* against uncertainty."
            ),
        }


class UCBAcquisition:
    """
    Upper Confidence Bound acquisition function.

        UCB(x) = μ(x) + κ · σ(x)

    Parameters
    ----------
    kappa : float
        Exploration weight κ. Larger κ = more exploration.
        κ=0 → pure exploitation, κ→∞ → pure uncertainty sampling.
    """

    name = "ucb"
    label = "Upper Confidence Bound"

    def __init__(self, kappa: float = 2.0, mode: str = "maximize", random_state: int = 42):
        self.kappa = kappa
        self.mode = mode
        self.random_state = random_state

    def score(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        X_selected: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        if self.mode == "maximize":
            return mean + self.kappa * std
        else:
            return -(mean - self.kappa * std)

    def select_batch(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_pool: np.ndarray,
        batch_size: int,
        already_selected: list | None = None,
        **kwargs,
    ) -> np.ndarray:
        excluded = set(already_selected or [])
        available = np.array([i for i in range(len(mean)) if i not in excluded])
        n_select = min(batch_size, len(available))
        scores = self.score(mean[available], std[available], X_pool[available])
        top_local = np.argsort(scores)[::-1][:n_select]
        return available[top_local]

    def breakdown(self, idx: int, mean: np.ndarray, std: np.ndarray,
                  X_pool: np.ndarray, **kwargs) -> dict:
        ucb_val = float(mean[idx] + self.kappa * std[idx])
        return {
            "strategy": self.label,
            "predicted_performance": float(mean[idx]),
            "uncertainty": float(std[idx]),
            "exploration_contribution": float(self.kappa * std[idx]),
            "diversity_contribution": 0.0,
            "final_score": ucb_val,
            "kappa": self.kappa,
            "selection_reason": (
                f"UCB={ucb_val:.4f} (μ={mean[idx]:.4f} + κ·σ={self.kappa*std[idx]:.4f}). "
                f"κ={self.kappa} weights uncertainty in the acquisition score."
            ),
        }


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - mn) / (mx - mn)
