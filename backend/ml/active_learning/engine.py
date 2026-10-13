"""
Active Learning Engine — the core decision loop.

This is the central component of the entire project. It orchestrates:

    Initial labeled set
            ↓
    Train surrogate model
            ↓
    Predict unlabeled pool  →  (μ, σ) per candidate
            ↓
    Calculate acquisition scores
            ↓
    Select candidates (with diversity)
            ↓
    Acquire results via simulator (reveals true dataset values)
            ↓
    Add to labeled set
            ↓
    Retrain model
            ↓
    Evaluate on held-out test set
            ↓
    Repeat

Design principles
-----------------
  1. Stateless between iterations: all state is in ExperimentState
  2. Strategy-agnostic: any AcquisitionStrategy can be plugged in
  3. No label leakage: the scaler is re-fit ONLY on labeled_X at each step
  4. Reproducible: random seed is recorded and honored throughout
  5. Transparent: every selection includes a full breakdown
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.ml.data.preprocessing import (
    PreprocessingConfig,
    PreprocessingState,
    fit_preprocessor,
)
from backend.ml.models.baseline import BaseSurrogate, RandomForestSurrogate
from backend.ml.models import bayesian
from backend.ml.models.bayesian import GaussianProcessSurrogate

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Full configuration for one active learning experiment run."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: str = "puda"           # random | uncertainty | ei | puda
    model_type: str = "gp"           # gp | rf | xgb | ensemble
    initial_samples: int = 100
    batch_size: int = 20
    total_budget: int = 300
    random_seed: int = 42
    # Model hyperparameters
    n_restarts_optimizer: int = 3    # GP only
    n_estimators: int = 200          # RF only
    model_params: Dict[str, Any] = field(default_factory=dict)
    # Acquisition hyperparameters
    xi: float = 0.01                 # EI exploration jitter
    kappa: float = 2.0               # UCB exploration weight
    alpha: float = 0.40              # PUDA prediction weight
    beta: float = 0.35               # PUDA uncertainty weight
    gamma: float = 0.25              # PUDA diversity weight
    # Preprocessing
    use_pca: bool = False
    pca_n_components: int = 50
    # Dataset info (set during initialization)
    dataset_id: Optional[str] = None
    target_col: str = "gap expt"
    feature_columns: List[str] = field(default_factory=list)


@dataclass
class CandidatePrediction:
    """Full prediction record for one candidate at one iteration."""
    candidate_id: str
    pool_index: int
    predicted_mean: float
    predicted_std: float
    lower_cb: Optional[float]
    upper_cb: Optional[float]
    acquisition_score: float
    is_selected: bool
    selection_rank: Optional[int]
    breakdown: Dict[str, Any]


@dataclass
class IterationResult:
    """Complete output of one active learning iteration."""
    iteration: int
    experiment_id: str
    strategy: str
    model_type: str
    # Budget
    budget_used: int
    budget_remaining: int
    # Selections
    selected_candidate_ids: List[str]
    selected_true_values: List[float]
    # Model performance (evaluated on test set)
    test_mae: float
    test_rmse: float
    test_r2: float
    # Discovery tracking
    best_found_value: float
    best_found_id: str
    regret: float
    global_maximum: float
    training_pool_maximum: float
    # Batch statistics
    mean_uncertainty_selected: float
    mean_score_selected: float
    diversity_score: float
    # Full pool predictions (for UI)
    all_predictions: List[CandidatePrediction]
    # Timing
    iteration_time_seconds: float
    timestamp: float


@dataclass
class ExperimentState:
    """Complete mutable state of an active learning experiment."""
    config: ExperimentConfig
    # Data (raw DataFrames before preprocessing)
    pool_X: pd.DataFrame = field(default_factory=pd.DataFrame)
    pool_y: pd.Series = field(default_factory=pd.Series)
    pool_ids: List[str] = field(default_factory=list)
    test_X: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_y: pd.Series = field(default_factory=pd.Series)
    test_ids: List[str] = field(default_factory=list)
    # Labeled / unlabeled tracking
    labeled_indices: List[int] = field(default_factory=list)    # indices into pool
    acquired_values: Dict[int, float] = field(default_factory=dict)  # pool_idx → true_y
    # History
    iterations: List[IterationResult] = field(default_factory=list)
    # Current preprocessor state
    preprocessor: Optional[PreprocessingState] = None
    # Current model
    model: Optional[BaseSurrogate] = None
    # Retrospective full-dataset reference maximum used for regret.
    global_maximum: float = 0.0
    # Maximum in the training pool; test rows remain unavailable to selection.
    training_pool_maximum: float = 0.0
    # Status
    is_initialized: bool = False
    current_iteration: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Active Learning Engine
# ─────────────────────────────────────────────────────────────────────────────

class ActiveLearningEngine:
    """
    Orchestrates the full active learning loop for materials discovery.

    Usage
    -----
        engine = ActiveLearningEngine(config)
        state  = engine.initialize(pool_X, pool_y, pool_ids, test_X, test_y, test_ids)

        for _ in range(n_iterations):
            result, state = engine.run_iteration(state)

        summary = engine.get_summary(state)
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._acquisition = self._build_acquisition(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize(
        self,
        pool_X: pd.DataFrame,
        pool_y: pd.Series,
        pool_ids: List[str],
        test_X: pd.DataFrame,
        test_y: pd.Series,
        test_ids: List[str],
        seed_indices: Optional[List[int]] = None,
        dataset_maximum: Optional[float] = None,
    ) -> ExperimentState:
        """
        Create the initial experiment state with a labeled seed set.

        Parameters
        ----------
        seed_indices : list, optional
            Specific pool indices to use as initial labeled set.
            If None, random sampling is used.
        """
        state = ExperimentState(config=self.config)
        state.pool_X = pool_X.reset_index(drop=True)
        state.pool_y = pool_y.reset_index(drop=True)
        state.pool_ids = list(pool_ids)
        state.test_X = test_X.reset_index(drop=True)
        state.test_y = test_y.reset_index(drop=True)
        state.test_ids = list(test_ids)
        state.training_pool_maximum = float(pool_y.max())
        state.global_maximum = float(
            dataset_maximum if dataset_maximum is not None
            else max(pool_y.max(), test_y.max())
        )

        # Select initial labeled set
        if seed_indices is not None:
            labeled_idx = list(seed_indices)
        else:
            rng = np.random.RandomState(self.config.random_seed)
            n_initial = min(self.config.initial_samples, len(pool_X) - 1)
            labeled_idx = list(rng.choice(len(pool_X), size=n_initial, replace=False))

        state.labeled_indices = labeled_idx
        # Record true values for initially labeled samples
        for idx in labeled_idx:
            state.acquired_values[idx] = float(pool_y.iloc[idx])

        state.is_initialized = True
        state.current_iteration = 0

        logger.info(
            "[Engine] Initialized | strategy=%s model=%s | "
            "seed=%d samples | pool=%d | test=%d | global_max=%.4f",
            self.config.strategy,
            self.config.model_type,
            len(labeled_idx),
            len(pool_X),
            len(test_X),
            state.global_maximum,
        )
        return state

    def train_model(self, state: ExperimentState) -> Tuple[BaseSurrogate, PreprocessingState]:
        """
        Fit preprocessor + surrogate model on the current labeled set.

        CRITICAL: scaler is fit ONLY on labeled_X (not on pool or test).
        """
        labeled_X = state.pool_X.iloc[state.labeled_indices]
        labeled_y = np.array([state.acquired_values[i] for i in state.labeled_indices])

        # Fit preprocessor on labeled set only
        prep_config = PreprocessingConfig(
            random_state=self.config.random_seed,
            use_pca=self.config.use_pca,
            pca_n_components=self.config.pca_n_components,
        )
        preprocessor = fit_preprocessor(
            labeled_X, prep_config, labeled_set_size=len(labeled_X)
        )

        X_train = preprocessor.transform(labeled_X)
        model = self._create_model()
        model.fit(X_train, labeled_y)

        logger.info(
            "[Engine] Model trained on %d labeled samples", len(labeled_y)
        )
        return model, preprocessor

    def predict_pool(
        self,
        state: ExperimentState,
        model: BaseSurrogate,
        preprocessor: PreprocessingState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict ALL pool candidates (labeled + unlabeled).

        Returns
        -------
        pool_mean, pool_std : predictions for all pool candidates
        unlabeled_idx : pool indices of the unlabeled candidates
        labeled_set : set of labeled pool indices
        """
        labeled_set = set(state.labeled_indices)
        unlabeled_idx = np.array([
            i for i in range(len(state.pool_X)) if i not in labeled_set
        ])

        if len(unlabeled_idx) == 0:
            return np.array([]), np.array([]), unlabeled_idx, labeled_set

        X_unlabeled = preprocessor.transform(state.pool_X.iloc[unlabeled_idx])
        mean, std = model.predict(X_unlabeled)

        return mean, std, unlabeled_idx, labeled_set

    def select_candidates(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        X_unlabeled: np.ndarray,
        unlabeled_global_idx: np.ndarray,
        y_labeled: np.ndarray,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply acquisition strategy to select the next batch.

        Returns
        -------
        selected_global_idx : pool-level indices of selected candidates
        acquisition_scores : scores for all unlabeled candidates
        """
        from backend.ml.acquisition.expected_improvement import ExpectedImprovementAcquisition

        # For EI, set the best observed value
        if hasattr(self._acquisition, "set_best_observed"):
            self._acquisition.set_best_observed(y_labeled)

        # Strategy-specific batch selection
        if self.config.strategy == "random":
            local_selected = self._acquisition.select_batch(
                mean, std, X_unlabeled, batch_size
            )
            scores = self._acquisition.score(mean, std, X_unlabeled)
        elif self.config.strategy == "uncertainty":
            local_selected = self._acquisition.select_batch(
                mean, std, X_unlabeled, batch_size
            )
            scores = self._acquisition.score(mean, std, X_unlabeled)
        elif self.config.strategy in ("ei", "ucb"):
            local_selected = self._acquisition.select_batch(
                mean, std, X_unlabeled, batch_size, y_labeled=y_labeled
            )
            scores = self._acquisition.score(
                mean, std, X_unlabeled, y_labeled=y_labeled
            )
        elif self.config.strategy == "puda":
            local_selected = self._acquisition.select_batch(
                mean, std, X_unlabeled, batch_size
            )
            scores = self._acquisition.score(mean, std, X_unlabeled)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        selected_global_idx = unlabeled_global_idx[local_selected]
        return selected_global_idx, scores

    def acquire_results(
        self,
        selected_global_idx: np.ndarray,
        state: ExperimentState,
    ) -> List[Tuple[int, str, float]]:
        """
        Simulate experimental evaluation by revealing true dataset values.

        This is a RETROSPECTIVE BENCHMARK: the true property value is read
        from the held-aside dataset (not generated or interpolated).

        Returns
        -------
        list of (pool_index, candidate_id, true_value)
        """
        results = []
        for idx in selected_global_idx:
            candidate_id = state.pool_ids[idx]
            true_value = float(state.pool_y.iloc[idx])
            results.append((int(idx), candidate_id, true_value))
            logger.debug(
                "[Simulator] Acquired %s → true value: %.4f eV",
                candidate_id, true_value
            )
        return results

    def update_dataset(
        self,
        state: ExperimentState,
        acquired: List[Tuple[int, str, float]],
    ) -> ExperimentState:
        """Add newly acquired observations to the labeled set."""
        for pool_idx, candidate_id, true_value in acquired:
            if pool_idx not in set(state.labeled_indices):
                state.labeled_indices.append(pool_idx)
                state.acquired_values[pool_idx] = true_value
        return state

    def evaluate_iteration(
        self,
        state: ExperimentState,
        model: BaseSurrogate,
        preprocessor: PreprocessingState,
    ) -> dict:
        """Compute test-set metrics for the current iteration."""
        X_test = preprocessor.transform(state.test_X)
        metrics = model.evaluate(X_test, state.test_y.values)

        # Best-found tracking
        best_val = float(max(state.acquired_values.values()))
        best_idx = max(state.acquired_values, key=state.acquired_values.get)
        best_id = state.pool_ids[best_idx]
        regret = float(state.global_maximum - best_val)

        # Diversity of selected candidates in this batch
        # (computed during run_iteration, stored separately)
        metrics.update({
            "best_found_value": best_val,
            "best_found_id": best_id,
            "regret": regret,
            "global_maximum": state.global_maximum,
            "training_pool_maximum": state.training_pool_maximum,
            "budget_used": len(state.labeled_indices),
            "budget_remaining": self.config.total_budget - len(state.labeled_indices),
        })
        return metrics

    def run_iteration(self, state: ExperimentState) -> Tuple[IterationResult, ExperimentState]:
        """
        Execute one complete active learning iteration.

        Returns the full IterationResult and updated ExperimentState.
        """
        t_start = time.time()

        if not state.is_initialized:
            raise RuntimeError("Call initialize() before run_iteration().")

        budget_used = len(state.labeled_indices)
        if budget_used >= self.config.total_budget:
            raise RuntimeError(
                f"Budget exhausted ({budget_used}/{self.config.total_budget}). "
                "No more iterations possible."
            )

        state.current_iteration += 1
        iteration = state.current_iteration
        logger.info(
            "[Engine] ── Iteration %d | budget %d/%d ──",
            iteration, budget_used, self.config.total_budget,
        )

        # 1. Train model
        model, preprocessor = self.train_model(state)
        state.model = model
        state.preprocessor = preprocessor

        # 2. Predict pool
        mean, std, unlabeled_idx, _ = self.predict_pool(state, model, preprocessor)
        if len(unlabeled_idx) == 0:
            raise RuntimeError("No unlabeled candidates remaining in pool.")

        X_unlabeled = preprocessor.transform(state.pool_X.iloc[unlabeled_idx])
        labeled_y = np.array([state.acquired_values[i] for i in state.labeled_indices])

        # 3. Select candidates
        remaining_budget = self.config.total_budget - budget_used
        actual_batch_size = min(self.config.batch_size, remaining_budget, len(unlabeled_idx))

        selected_global_idx, acq_scores = self.select_candidates(
            mean, std, X_unlabeled, unlabeled_idx, labeled_y, actual_batch_size
        )

        # 4. Acquire results (simulator)
        acquired = self.acquire_results(selected_global_idx, state)

        # 5. Update dataset
        state = self.update_dataset(state, acquired)

        # 6. Evaluate on test set
        eval_metrics = self.evaluate_iteration(state, model, preprocessor)

        # 7. Build per-candidate prediction records
        selected_set = set(selected_global_idx.tolist())
        selected_ranks = {int(idx): rank + 1
                         for rank, idx in enumerate(selected_global_idx)}

        all_predictions = []
        for local_i, pool_i in enumerate(unlabeled_idx):
            μ = float(mean[local_i])
            σ = float(std[local_i])
            score = float(acq_scores[local_i]) if local_i < len(acq_scores) else 0.0
            is_sel = pool_i in selected_set

            # Get breakdown for selected candidates
            breakdown: dict = {}
            if is_sel and hasattr(self._acquisition, "breakdown"):
                try:
                    breakdown = self._acquisition.breakdown(
                        local_i, mean, std, X_unlabeled
                    )
                except Exception:
                    breakdown = {}

            interval_multiplier = bayesian.CI_MULTIPLIER
            calibrated_interval = isinstance(model, GaussianProcessSurrogate)
            all_predictions.append(CandidatePrediction(
                candidate_id=state.pool_ids[pool_i],
                pool_index=int(pool_i),
                predicted_mean=μ,
                predicted_std=σ,
                lower_cb=μ - interval_multiplier * σ if calibrated_interval else None,
                upper_cb=μ + interval_multiplier * σ if calibrated_interval else None,
                acquisition_score=score,
                is_selected=is_sel,
                selection_rank=selected_ranks.get(pool_i),
                breakdown=breakdown,
            ))

        # 8. Compute batch diversity score
        diversity = self._compute_batch_diversity(
            X_unlabeled, [np.where(unlabeled_idx == g)[0][0]
                          for g in selected_global_idx
                          if g in unlabeled_idx]
        )

        result = IterationResult(
            iteration=iteration,
            experiment_id=self.config.experiment_id,
            strategy=self.config.strategy,
            model_type=self.config.model_type,
            budget_used=len(state.labeled_indices),
            budget_remaining=self.config.total_budget - len(state.labeled_indices),
            selected_candidate_ids=[cid for _, cid, _ in acquired],
            selected_true_values=[val for _, _, val in acquired],
            test_mae=eval_metrics["mae"],
            test_rmse=eval_metrics["rmse"],
            test_r2=eval_metrics["r2"],
            best_found_value=eval_metrics["best_found_value"],
            best_found_id=eval_metrics["best_found_id"],
            regret=eval_metrics["regret"],
            global_maximum=eval_metrics["global_maximum"],
            training_pool_maximum=eval_metrics["training_pool_maximum"],
            mean_uncertainty_selected=float(
                std[[np.where(unlabeled_idx == g)[0][0]
                     for g in selected_global_idx if g in unlabeled_idx]].mean()
                if len(selected_global_idx) > 0 else 0.0
            ),
            mean_score_selected=float(
                np.mean([acq_scores[np.where(unlabeled_idx == g)[0][0]]
                         for g in selected_global_idx if g in unlabeled_idx])
                if len(selected_global_idx) > 0 else 0.0
            ),
            diversity_score=diversity,
            all_predictions=all_predictions,
            iteration_time_seconds=time.time() - t_start,
            timestamp=time.time(),
        )

        state.iterations.append(result)

        logger.info(
            "[Engine] Iteration %d done | RMSE=%.4f R²=%.4f | "
            "best=%.4f (regret=%.4f) | time=%.1fs",
            iteration,
            result.test_rmse, result.test_r2,
            result.best_found_value, result.regret,
            result.iteration_time_seconds,
        )

        return result, state

    def get_summary(self, state: ExperimentState) -> dict:
        """Return a compact experiment summary for the overview page."""
        if not state.iterations:
            return {"status": "no_iterations"}

        last = state.iterations[-1]
        return {
            "experiment_id": self.config.experiment_id,
            "strategy": self.config.strategy,
            "model_type": self.config.model_type,
            "total_iterations": len(state.iterations),
            "budget_used": last.budget_used,
            "budget_total": self.config.total_budget,
            "budget_pct": round(100 * last.budget_used / self.config.total_budget, 1),
            "best_found_value": last.best_found_value,
            "best_found_id": last.best_found_id,
            "final_test_rmse": last.test_rmse,
            "final_test_r2": last.test_r2,
            "final_regret": last.regret,
            "global_maximum": last.global_maximum,
            "rmse_curve": [r.test_rmse for r in state.iterations],
            "r2_curve": [r.test_r2 for r in state.iterations],
            "best_found_curve": [r.best_found_value for r in state.iterations],
            "regret_curve": [r.regret for r in state.iterations],
            "budget_curve": [r.budget_used for r in state.iterations],
            "diversity_curve": [r.diversity_score for r in state.iterations],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_acquisition(self, config: ExperimentConfig):
        """Instantiate the correct acquisition strategy from config."""
        from backend.ml.acquisition.random import RandomAcquisition
        from backend.ml.acquisition.uncertainty import UncertaintyAcquisition
        from backend.ml.acquisition.expected_improvement import (
            ExpectedImprovementAcquisition, UCBAcquisition
        )
        from backend.ml.acquisition.hybrid_puda import PUDAAcquisition

        registry = {
            "random": lambda: RandomAcquisition(random_state=config.random_seed),
            "uncertainty": lambda: UncertaintyAcquisition(random_state=config.random_seed),
            "ei": lambda: ExpectedImprovementAcquisition(
                xi=config.xi, random_state=config.random_seed
            ),
            "ucb": lambda: UCBAcquisition(
                kappa=config.kappa, random_state=config.random_seed
            ),
            "puda": lambda: PUDAAcquisition(
                alpha=config.alpha,
                beta=config.beta,
                gamma=config.gamma,
                random_state=config.random_seed,
            ),
        }
        if config.strategy not in registry:
            raise ValueError(
                f"Unknown strategy '{config.strategy}'. "
                f"Available: {list(registry.keys())}"
            )
        return registry[config.strategy]()

    def _create_model(self) -> BaseSurrogate:
        """Create surrogate model from config."""
        from backend.ml.models.bayesian import create_bayesian_surrogate
        from backend.ml.models.baseline import create_surrogate

        if self.config.model_type in ("gp", "gaussian_process", "ensemble", "deep_ensemble"):
            model_kwargs = {"random_state": self.config.random_seed}
            if self.config.model_type in ("gp", "gaussian_process"):
                model_kwargs["n_restarts_optimizer"] = self.config.n_restarts_optimizer
                model_kwargs.update(self.config.model_params)
            return create_bayesian_surrogate(self.config.model_type, **model_kwargs)
        else:
            model_kwargs = {
                "n_estimators": self.config.n_estimators,
                **self.config.model_params,
            }
            return create_surrogate(
                self.config.model_type,
                random_state=self.config.random_seed,
                **model_kwargs,
            )

    def _compute_batch_diversity(
        self,
        X_pool: np.ndarray,
        selected_local_indices: list,
    ) -> float:
        """
        Compute mean pairwise cosine distance within the selected batch.
        Returns 0.0 if batch has fewer than 2 candidates.
        """
        if len(selected_local_indices) < 2:
            return 0.0
        from sklearn.metrics.pairwise import cosine_distances
        X_sel = X_pool[selected_local_indices]
        dists = cosine_distances(X_sel, X_sel)
        # Upper triangle only (exclude diagonal)
        n = len(X_sel)
        upper = dists[np.triu_indices(n, k=1)]
        return float(upper.mean()) if len(upper) > 0 else 0.0
