"""
Multi-strategy comparison framework.

Aggregates results from experiments run with different acquisition strategies
and produces:
  - Side-by-side performance tables
  - Discovery curves per strategy
  - RMSE vs budget curves
  - Regret curves
  - Diversity curves
  - Statistical summary (NAUDC ranking)
  - Ablation study comparison

Used by the Strategy Comparison page (Page 8) in the frontend.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from backend.ml.evaluation.metrics import (
    best_found_curve,
    cumulative_regret,
    normalized_area_under_discovery_curve,
    top_k_discovery_rate,
)

logger = logging.getLogger(__name__)

STRATEGY_LABELS = {
    "random": "Random Sampling",
    "uncertainty": "Uncertainty Sampling",
    "ei": "Expected Improvement",
    "ucb": "Upper Confidence Bound",
    "puda": "PUDA (Proposed)",
    "puda_pred_only": "PUDA — Pred Only",
    "puda_pred_unc": "PUDA — Pred + Unc",
    "puda_pred_div": "PUDA — Pred + Div",
    "puda_full": "PUDA — Full",
}

STRATEGY_COLORS = {
    "random": "#94a3b8",
    "uncertainty": "#60a5fa",
    "ei": "#34d399",
    "ucb": "#a78bfa",
    "puda": "#f97316",
    "puda_pred_only": "#fca5a5",
    "puda_pred_unc": "#fdba74",
    "puda_pred_div": "#fde68a",
    "puda_full": "#f97316",
}


class StrategyComparison:
    """
    Collects experiment results from multiple strategies and builds
    all comparison charts/tables needed by the frontend.

    Usage
    -----
        comp = StrategyComparison(global_optimum=12.0, mode="maximize")
        comp.add_experiment("random", ...)
        comp.add_experiment("puda", ...)
        charts = comp.build_charts()
    """

    def __init__(self, global_optimum: float, mode: str = "maximize"):
        self.global_optimum = global_optimum
        self.mode = mode
        # {strategy_name: experiment_data_dict}
        self._experiments: Dict[str, dict] = {}

    def add_experiment(
        self,
        strategy: str,
        budget_curve: List[int],
        rmse_curve: List[float],
        r2_curve: List[float],
        best_found_curve_vals: List[float],
        acquired_values_in_order: List[float],
        diversity_curve: List[float],
        all_pool_values: np.ndarray,
    ) -> None:
        """
        Register one strategy's results.

        Parameters
        ----------
        strategy : strategy key (e.g. 'puda')
        budget_curve : list of budget_used values per iteration
        rmse_curve : test RMSE per iteration
        r2_curve : test R² per iteration
        best_found_curve_vals : best-so-far property per iteration
        acquired_values_in_order : true values in acquisition order
        diversity_curve : mean batch diversity per iteration
        all_pool_values : all pool true values (for top-k reference)
        """
        regret_c = cumulative_regret(best_found_curve_vals, self.global_optimum, self.mode)
        naudc = normalized_area_under_discovery_curve(
            best_found_curve_vals, self.global_optimum, self.mode
        )
        final_regret = regret_c[-1] if regret_c else float(self.global_optimum)
        final_rmse = rmse_curve[-1] if rmse_curve else float("inf")
        final_r2 = r2_curve[-1] if r2_curve else 0.0
        top10 = top_k_discovery_rate(
            acquired_values_in_order, all_pool_values, k=10, mode=self.mode
        )

        self._experiments[strategy] = {
            "strategy": strategy,
            "label": STRATEGY_LABELS.get(strategy, strategy),
            "color": STRATEGY_COLORS.get(strategy, "#888888"),
            "budget_curve": budget_curve,
            "rmse_curve": rmse_curve,
            "r2_curve": r2_curve,
            "best_found_curve": best_found_curve_vals,
            "regret_curve": regret_c,
            "diversity_curve": diversity_curve,
            "naudc": naudc,
            "final_regret": final_regret,
            "final_rmse": final_rmse,
            "final_r2": final_r2,
            "top10_discovery_rate": top10,
            "n_iterations": len(budget_curve),
        }
        logger.info(
            "[Comparison] Registered strategy=%s | NAUDC=%.4f | "
            "final_regret=%.4f | final_RMSE=%.4f",
            strategy, naudc, final_regret, final_rmse,
        )

    def build_charts(self) -> dict:
        """
        Build all chart data structures for the frontend.

        Returns a dict with keys:
            discovery_curve, rmse_curve, regret_curve, diversity_curve,
            summary_table, ranking
        """
        if not self._experiments:
            return {}

        # Common budget axis (use longest experiment's budget curve)
        max_iters = max(len(v["budget_curve"]) for v in self._experiments.values())

        def pad(lst, length, fill=None):
            """Right-pad a list to `length` with last value or fill."""
            if fill is None and lst:
                fill = lst[-1]
            return lst + [fill] * (length - len(lst)) if len(lst) < length else lst

        strategies = list(self._experiments.keys())

        # ── Discovery curve ──────────────────────────────────────────────────
        discovery_series = []
        for s in strategies:
            exp = self._experiments[s]
            discovery_series.append({
                "strategy": s,
                "label": exp["label"],
                "color": exp["color"],
                "budget": exp["budget_curve"],
                "best_found": exp["best_found_curve"],
            })

        # ── RMSE curve ───────────────────────────────────────────────────────
        rmse_series = []
        for s in strategies:
            exp = self._experiments[s]
            rmse_series.append({
                "strategy": s,
                "label": exp["label"],
                "color": exp["color"],
                "budget": exp["budget_curve"],
                "rmse": exp["rmse_curve"],
            })

        # ── Regret curve ─────────────────────────────────────────────────────
        regret_series = []
        for s in strategies:
            exp = self._experiments[s]
            regret_series.append({
                "strategy": s,
                "label": exp["label"],
                "color": exp["color"],
                "budget": exp["budget_curve"],
                "regret": exp["regret_curve"],
            })

        # ── Diversity curve ──────────────────────────────────────────────────
        diversity_series = []
        for s in strategies:
            exp = self._experiments[s]
            diversity_series.append({
                "strategy": s,
                "label": exp["label"],
                "color": exp["color"],
                "budget": exp["budget_curve"],
                "diversity": exp["diversity_curve"],
            })

        # ── Summary table ────────────────────────────────────────────────────
        summary_table = []
        for s in strategies:
            exp = self._experiments[s]
            summary_table.append({
                "strategy": s,
                "label": exp["label"],
                "color": exp["color"],
                "naudc": round(exp["naudc"], 4),
                "final_rmse": round(exp["final_rmse"], 4),
                "final_r2": round(exp["final_r2"], 4),
                "final_regret": round(exp["final_regret"], 4),
                "top10_discovery_rate": round(exp["top10_discovery_rate"], 3),
                "n_iterations": exp["n_iterations"],
            })

        # Sort by NAUDC descending
        summary_table.sort(key=lambda r: r["naudc"], reverse=True)
        for rank, row in enumerate(summary_table):
            row["rank"] = rank + 1

        # ── Ranking ──────────────────────────────────────────────────────────
        ranking = {
            "by_naudc": [r["strategy"] for r in summary_table],
            "best_strategy_naudc": summary_table[0]["strategy"] if summary_table else None,
            "best_strategy_final_rmse": min(strategies, key=lambda s: self._experiments[s]["final_rmse"]),
            "best_strategy_final_regret": min(strategies, key=lambda s: self._experiments[s]["final_regret"]),
        }

        return {
            "discovery_curve": discovery_series,
            "rmse_curve": rmse_series,
            "regret_curve": regret_series,
            "diversity_curve": diversity_series,
            "summary_table": summary_table,
            "ranking": ranking,
            "global_optimum": self.global_optimum,
            "strategies": strategies,
        }

    def ablation_summary(self) -> dict:
        """
        Build ablation study comparison for PUDA variants.
        Expects strategies like 'puda_pred_only', 'puda_pred_unc', 'puda_full'.
        """
        ablation_keys = [s for s in self._experiments if s.startswith("puda")]
        if not ablation_keys:
            return {}

        rows = []
        for s in ablation_keys:
            exp = self._experiments[s]
            rows.append({
                "variant": s,
                "label": exp["label"],
                "naudc": round(exp["naudc"], 4),
                "final_rmse": round(exp["final_rmse"], 4),
                "final_r2": round(exp["final_r2"], 4),
                "final_regret": round(exp["final_regret"], 4),
            })

        rows.sort(key=lambda r: r["naudc"], reverse=True)
        return {"ablation_table": rows, "best_variant": rows[0]["variant"] if rows else None}


def build_comparison_from_db_records(records: list) -> dict:
    """
    Build comparison charts from a list of DB experiment records.

    Each record is a dict with:
        strategy, budget_curve, rmse_curve, r2_curve,
        best_found_curve, diversity_curve, acquired_values_in_order,
        global_maximum

    Suitable for use in the evaluation API endpoint.
    """
    if not records:
        return {}

    global_optimum = max(r.get("global_maximum", 0) for r in records)
    comp = StrategyComparison(global_optimum=global_optimum)

    for rec in records:
        all_pool_values = np.array(rec.get("all_pool_values", []))
        comp.add_experiment(
            strategy=rec["strategy"],
            budget_curve=rec.get("budget_curve", []),
            rmse_curve=rec.get("rmse_curve", []),
            r2_curve=rec.get("r2_curve", []),
            best_found_curve_vals=rec.get("best_found_curve", []),
            acquired_values_in_order=rec.get("acquired_values_in_order", []),
            diversity_curve=rec.get("diversity_curve", []),
            all_pool_values=all_pool_values,
        )

    return comp.build_charts()
