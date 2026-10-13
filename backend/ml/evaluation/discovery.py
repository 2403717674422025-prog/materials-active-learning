"""
Material discovery tracking and ranked results.

Tracks which materials have been discovered across all iterations,
their true property values, which strategy found them and when,
and builds the ranked discovery table shown on Page 9 (Material Discovery).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredMaterial:
    """Record for a single discovered material."""
    rank: int
    candidate_id: str
    true_value: float
    iteration_discovered: int
    strategy: str
    budget_at_discovery: int
    predicted_mean: float
    predicted_std: float
    acquisition_score: float
    # For display: feature values (subset of most important features)
    feature_preview: Dict[str, float] = field(default_factory=dict)


class DiscoveryTracker:
    """
    Tracks the best materials discovered across all active learning iterations.

    Maintains a sorted list of discovered materials and provides:
    - Ranked discovery table
    - Best-found value at each budget checkpoint
    - Which iteration each top-k material was first found
    - Comparison across strategies
    """

    def __init__(self, global_optimum: float, mode: str = "maximize"):
        self.global_optimum = global_optimum
        self.mode = mode
        self._discovered: Dict[str, DiscoveredMaterial] = {}  # id → record
        self._acquisition_order: List[tuple] = []  # (iteration, candidate_id, true_value)

    def record_acquisition(
        self,
        iteration: int,
        candidate_id: str,
        true_value: float,
        strategy: str,
        budget_used: int,
        predicted_mean: float = 0.0,
        predicted_std: float = 0.0,
        acquisition_score: float = 0.0,
        feature_preview: Optional[dict] = None,
    ) -> None:
        """Record that a material was acquired in this iteration."""
        self._acquisition_order.append((iteration, candidate_id, true_value))

        # Only record if not already discovered (first discovery wins)
        if candidate_id not in self._discovered:
            self._discovered[candidate_id] = DiscoveredMaterial(
                rank=0,  # recalculated in get_ranked_list
                candidate_id=candidate_id,
                true_value=float(true_value),
                iteration_discovered=iteration,
                strategy=strategy,
                budget_at_discovery=budget_used,
                predicted_mean=float(predicted_mean),
                predicted_std=float(predicted_std),
                acquisition_score=float(acquisition_score),
                feature_preview=feature_preview or {},
            )

    def get_ranked_list(self, top_n: Optional[int] = None) -> List[DiscoveredMaterial]:
        """
        Return discovered materials sorted by true_value.
        Assigns rank based on sorted order.
        """
        materials = list(self._discovered.values())
        reverse = self.mode == "maximize"
        materials.sort(key=lambda m: m.true_value, reverse=reverse)

        for rank, mat in enumerate(materials):
            mat.rank = rank + 1

        if top_n:
            return materials[:top_n]
        return materials

    def best_found_at_budget(self, budget: int) -> Optional[float]:
        """Best discovered value up to a given budget."""
        subset = [v for _, _, v in self._acquisition_order
                  if len(self._acquisition_order[:self._acquisition_order.index(
                      (next((t for t in self._acquisition_order if t[0] <= budget),
                             self._acquisition_order[-1] if self._acquisition_order else None),
                       None, None)
                  )]) <= budget]
        if not subset:
            return None
        return max(subset) if self.mode == "maximize" else min(subset)

    def get_acquisition_order_values(self) -> List[float]:
        """Return true values in acquisition order."""
        return [v for _, _, v in self._acquisition_order]

    def first_discovery_iteration(self, top_k: int) -> Dict[str, int]:
        """
        For the true top-k materials, report the iteration they were first acquired.

        Returns
        -------
        dict mapping candidate_id → iteration_discovered (None if not discovered)
        """
        ranked = self.get_ranked_list(top_n=top_k)
        return {
            mat.candidate_id: mat.iteration_discovered
            for mat in ranked
        }

    def summary_stats(self) -> dict:
        """Quick summary statistics for the experiment overview page."""
        if not self._discovered:
            return {"n_discovered": 0}

        values = [m.true_value for m in self._discovered.values()]
        best = max(values) if self.mode == "maximize" else min(values)

        return {
            "n_discovered": len(self._discovered),
            "best_found_value": float(best),
            "best_found_regret": float(abs(self.global_optimum - best)),
            "mean_discovered_value": float(np.mean(values)),
            "std_discovered_value": float(np.std(values)),
        }

    def to_dict_list(self, top_n: Optional[int] = 50) -> List[dict]:
        """Serialize ranked list to JSON-safe list of dicts."""
        return [
            {
                "rank": m.rank,
                "candidate_id": m.candidate_id,
                "true_value": m.true_value,
                "iteration_discovered": m.iteration_discovered,
                "strategy": m.strategy,
                "budget_at_discovery": m.budget_at_discovery,
                "predicted_mean": m.predicted_mean,
                "predicted_std": m.predicted_std,
                "acquisition_score": m.acquisition_score,
                "feature_preview": m.feature_preview,
            }
            for m in self.get_ranked_list(top_n=top_n)
        ]
