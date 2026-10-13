"""
Evaluation metrics for the active learning system.

Covers:
  A. Predictive performance  — MAE, RMSE, R²
  B. Discovery efficiency    — best-found curve, top-k discovery rate
  C. Cumulative regret       — distance from global optimum
  D. Budget efficiency       — performance per unit of budget spent
  E. Diversity               — mean pairwise cosine distance in selections
  F. Calibration             — reliability of GP uncertainty estimates (ECE)
  G. NAUDC                   — Normalized Area Under Discovery Curve (summary scalar)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# A. Predictive performance
# ─────────────────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, R² on a hold-out set."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# B. Discovery efficiency
# ─────────────────────────────────────────────────────────────────────────────

def best_found_curve(
    acquisition_order: List[float],
    mode: str = "maximize",
) -> List[float]:
    """
    Compute the best-so-far value after each acquisition step.

    Parameters
    ----------
    acquisition_order : list of float
        True property values in the order they were acquired.
    mode : 'maximize' or 'minimize'

    Returns
    -------
    curve : list of float, same length as acquisition_order
        curve[i] = best value seen after acquiring i+1 samples.
    """
    if not acquisition_order:
        return []

    curve = []
    best = acquisition_order[0]
    for val in acquisition_order:
        if mode == "maximize":
            best = max(best, val)
        else:
            best = min(best, val)
        curve.append(best)
    return curve


def top_k_discovery_rate(
    acquired_values: List[float],
    all_values: np.ndarray,
    k: int,
    mode: str = "maximize",
) -> float:
    """
    Fraction of the true top-k materials that have been discovered.

    Parameters
    ----------
    acquired_values : values of acquired candidates so far
    all_values : true values for the entire pool
    k : how many top materials to track
    mode : 'maximize' or 'minimize'

    Returns
    -------
    rate : float in [0, 1]
    """
    if mode == "maximize":
        threshold = np.sort(all_values)[::-1][k - 1]
        n_found = sum(1 for v in acquired_values if v >= threshold)
    else:
        threshold = np.sort(all_values)[k - 1]
        n_found = sum(1 for v in acquired_values if v <= threshold)
    return min(n_found / k, 1.0)


def discovery_at_budget(
    acquired_values: List[float],
    all_pool_values: np.ndarray,
    budgets: List[int],
    mode: str = "maximize",
) -> dict:
    """
    Report best-found value and top-10 discovery rate at specific budget points.

    Parameters
    ----------
    acquired_values : all acquired values in acquisition order
    all_pool_values : ground truth for full pool (for top-k reference)
    budgets : budget checkpoints to report (e.g. [10, 20, 50, 100, 200])

    Returns
    -------
    dict mapping budget → {best_found, top10_rate, top5_rate}
    """
    results = {}
    for b in budgets:
        subset = acquired_values[:b]
        if not subset:
            results[b] = {"best_found": None, "top10_rate": 0.0, "top5_rate": 0.0}
            continue
        if mode == "maximize":
            best = max(subset)
        else:
            best = min(subset)
        top10 = top_k_discovery_rate(subset, all_pool_values, k=10, mode=mode)
        top5 = top_k_discovery_rate(subset, all_pool_values, k=5, mode=mode)
        results[b] = {
            "best_found": float(best),
            "top10_rate": float(top10),
            "top5_rate": float(top5),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# C. Cumulative regret
# ─────────────────────────────────────────────────────────────────────────────

def cumulative_regret(
    best_found_curve_vals: List[float],
    global_optimum: float,
    mode: str = "maximize",
) -> List[float]:
    """
    Instantaneous regret at each step = global_optimum - best_found.

    In a perfect strategy, regret → 0. Random sampling maintains high regret.
    """
    if mode == "maximize":
        return [float(global_optimum - v) for v in best_found_curve_vals]
    else:
        return [float(v - global_optimum) for v in best_found_curve_vals]


def simple_regret(
    acquired_values: List[float],
    global_optimum: float,
    mode: str = "maximize",
) -> float:
    """Final simple regret = global_optimum - best_value_found."""
    if not acquired_values:
        return float(global_optimum)
    if mode == "maximize":
        return float(global_optimum - max(acquired_values))
    else:
        return float(min(acquired_values) - global_optimum)


# ─────────────────────────────────────────────────────────────────────────────
# D. Budget efficiency (NAUDC)
# ─────────────────────────────────────────────────────────────────────────────

def normalized_area_under_discovery_curve(
    best_found_curve_vals: List[float],
    global_optimum: float,
    mode: str = "maximize",
    budgets: Optional[List[float]] = None,
) -> float:
    """
    Normalized Area Under the Discovery Curve (NAUDC).

    Normalize the best-found values by the global optimum and integrate over
    the supplied cumulative-candidate budgets using the trapezoidal rule.
    Divide by the budget span to obtain a time-averaged normalized value.
    When budgets are omitted, observations are equally spaced at 0..N-1.
    """
    if not best_found_curve_vals:
        return 0.0
    values = np.asarray(best_found_curve_vals, dtype=float)
    x = np.asarray(budgets if budgets is not None else range(len(values)), dtype=float)
    if len(x) != len(values):
        raise ValueError("budgets and best_found_curve_vals must have equal lengths")
    if not np.isfinite(values).all() or not np.isfinite(x).all():
        raise ValueError("curve values and budgets must be finite")
    if len(values) > 1 and np.any(np.diff(x) <= 0):
        raise ValueError("curve values and strictly increasing budgets must be finite")
    if global_optimum == 0:
        return 0.0
    if len(values) == 1:
        if mode == "maximize":
            return float(values[0] / global_optimum)
        return float(global_optimum / max(values[0], 1e-9))
    if mode == "maximize":
        normalized = values / global_optimum
    else:
        normalized = global_optimum / np.maximum(values, 1e-9)
    return float(np.trapz(normalized, x) / (x[-1] - x[0]))


def discovery_curve_with_initial(
    initial_best: float, acquired_values: List[float], initial_budget: int,
    mode: str = "maximize",
) -> tuple[List[int], List[float]]:
    """Return best-found points at the initial and each cumulative candidate budget."""
    values = [float(initial_best)]
    best = float(initial_best)
    budgets = [int(initial_budget)]
    for offset, value in enumerate(acquired_values, 1):
        best = max(best, float(value)) if mode == "maximize" else min(best, float(value))
        values.append(best)
        budgets.append(int(initial_budget) + offset)
    return budgets, values


# ─────────────────────────────────────────────────────────────────────────────
# E. Batch diversity
# ─────────────────────────────────────────────────────────────────────────────

def batch_diversity_score(X_selected: np.ndarray) -> float:
    """
    Mean pairwise cosine distance within the selected batch.

    Returns 0.0 if batch has fewer than 2 candidates.
    Range: [0, 2] for cosine distance (1 = orthogonal, 2 = opposite).
    Normalized to [0, 1] by dividing by 2.
    """
    if len(X_selected) < 2:
        return 0.0
    from sklearn.metrics.pairwise import cosine_distances
    dists = cosine_distances(X_selected, X_selected)
    n = len(X_selected)
    upper = dists[np.triu_indices(n, k=1)]
    return float(upper.mean() / 2.0) if len(upper) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# F. Uncertainty calibration
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Coverage-based empirical calibration error for regression intervals.

    ECE is the mean absolute difference between nominal Gaussian interval
    coverage and observed empirical coverage over ``n_bins`` confidence
    levels. It is a coverage diagnostic, not a complete test of probabilistic
    calibration.

    Returns
    -------
    dict with:
        ece : float — coverage-based empirical calibration error
        reliability_diagram : dict — for plotting
        coverage_90pct : fraction of true values within nominal 90% intervals
        coverage_95pct : fraction of true values within nominal 95% intervals
        n_observations : number of evaluated observations
    """
    from scipy.stats import norm

    arrays = []
    for name, values in (
        ("y_true", y_true),
        ("y_pred_mean", y_pred_mean),
        ("y_pred_std", y_pred_std),
    ):
        try:
            array = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric values.") from exc
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values.")
        arrays.append(array)

    y_true, y_pred_mean, y_pred_std = arrays
    if not (len(y_true) == len(y_pred_mean) == len(y_pred_std)):
        raise ValueError("y_true, y_pred_mean, and y_pred_std must have equal lengths.")
    if np.any(y_pred_std < 0):
        raise ValueError("y_pred_std must not contain negative values.")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    confidence_levels = np.linspace(0.05, 0.95, n_bins)
    observed_coverage = []

    for conf in confidence_levels:
        z = norm.ppf((1 + conf) / 2)
        within = np.abs(y_true - y_pred_mean) <= z * y_pred_std
        observed_coverage.append(float(within.mean()))

    ece = float(np.mean(np.abs(np.array(observed_coverage) - confidence_levels)))

    # Specific intervals
    z90 = norm.ppf(0.95)
    z95 = norm.ppf(0.975)
    cov90 = float((np.abs(y_true - y_pred_mean) <= z90 * y_pred_std).mean())
    cov95 = float((np.abs(y_true - y_pred_mean) <= z95 * y_pred_std).mean())

    return {
        "ece": ece,
        "coverage_90pct": cov90,
        "coverage_95pct": cov95,
        "n_observations": int(len(y_true)),
        "reliability_diagram": {
            "expected_coverage": confidence_levels.tolist(),
            "observed_coverage": observed_coverage,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# G. Per-iteration summary
# ─────────────────────────────────────────────────────────────────────────────

def compute_iteration_metrics(
    iteration: int,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_std: np.ndarray,
    acquired_values_in_order: List[float],
    global_optimum: float,
    all_pool_values: np.ndarray,
    budget_used: int,
    diversity: float,
    mode: str = "maximize",
) -> dict:
    """
    Compute all metrics for one AL iteration and return as a flat dict.
    Suitable for inserting directly into the evaluation_metrics DB table.
    """
    pred_metrics = regression_metrics(y_test_true, y_test_pred)

    curve = best_found_curve(acquired_values_in_order, mode=mode)
    regret_curve = cumulative_regret(curve, global_optimum, mode=mode)

    budgets = [10, 20, 50, 100, 200, 300, 500]
    disc = discovery_at_budget(acquired_values_in_order, all_pool_values, budgets, mode)

    naudc = normalized_area_under_discovery_curve(curve, global_optimum, mode)

    cal = expected_calibration_error(y_test_true, y_test_pred, y_test_std) \
        if y_test_std is not None and len(y_test_std) > 0 else {}

    return {
        "iteration": iteration,
        "budget_used": budget_used,
        # Predictive performance
        "mae": pred_metrics["mae"],
        "rmse": pred_metrics["rmse"],
        "r2": pred_metrics["r2"],
        # Discovery
        "best_found": float(curve[-1]) if curve else 0.0,
        "regret": float(regret_curve[-1]) if regret_curve else float(global_optimum),
        "naudc": naudc,
        # Calibration
        "ece": cal.get("ece", 0.0),
        "coverage_90pct": cal.get("coverage_90pct", 0.0),
        "coverage_95pct": cal.get("coverage_95pct", 0.0),
        # Diversity
        "diversity_score": diversity,
        # Discovery at budget checkpoints
        "discovery_at_budget": disc,
    }
