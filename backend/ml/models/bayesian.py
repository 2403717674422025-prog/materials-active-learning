"""
Gaussian Process Regression surrogate model.

Provides the primary Bayesian probabilistic model with exact posterior
predictive distributions:

    p(y* | x*, X, y) = N(μ(x*), σ²(x*))

where:
    μ(x*) = posterior mean (our prediction)
    σ(x*) = posterior standard deviation (our uncertainty estimate)

Kernel choice:
    k(x, x') = σ²_f · RBF(l) + σ²_n · WhiteKernel()

    - RBF captures smooth correlations between material descriptors
    - WhiteKernel models additive Gaussian observation noise
    - Hyperparameters are optimized by maximizing log-marginal-likelihood

Scalability:
    Standard GP has O(N³) training complexity. For N > GP_MAX_SAMPLES,
    the system logs a warning and recommends switching to the RF surrogate.
    PCA preprocessing (see preprocessing.py) extends practical usability
    to N ≈ 1,000–1,500.

Output format per candidate:
    {
        "predicted_property":    μ(x),
        "prediction_std":        σ(x),
        "lower_confidence_bound": μ(x) - 1.96 * σ(x),
        "upper_confidence_bound": μ(x) + 1.96 * σ(x),
    }
"""

from __future__ import annotations

import logging
import warnings
from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from backend.ml.models.baseline import BaseSurrogate

logger = logging.getLogger(__name__)

# Above this labeled-set size, warn about GP scalability
GP_MAX_SAMPLES = 1500
# 95% confidence interval multiplier
CI_MULTIPLIER = 1.96


class GaussianProcessSurrogate(BaseSurrogate):
    """
    Exact Gaussian Process Regressor with RBF + White noise kernel.

    Parameters
    ----------
    length_scale : float
        Initial length scale for the RBF kernel. Optimized during fit.
    noise_level : float
        Initial noise variance for the WhiteKernel. Optimized during fit.
    n_restarts_optimizer : int
        Number of random restarts for the log-marginal-likelihood optimizer.
        Higher values reduce risk of local optima at the cost of fit time.
    normalize_y : bool
        Subtract the mean of y during training. Recommended for GP stability.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        noise_level: float = 0.1,
        n_restarts_optimizer: int = 3,
        normalize_y: bool = True,
        random_state: int = 42,
    ):
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state
        self._model: GaussianProcessRegressor | None = None
        self._n_train: int = 0

    def _build_kernel(self):
        """Build composite kernel: ConstantKernel * RBF + WhiteKernel."""
        amplitude = ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(1e-3, 1e3),
        )
        rbf = RBF(
            length_scale=self.length_scale,
            length_scale_bounds=(1e-3, 1e3),
        )
        white = WhiteKernel(
            noise_level=self.noise_level,
            noise_level_bounds=(1e-5, 1e1),
        )
        return amplitude * rbf + white

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        n = X.shape[0]
        self._n_train = n

        if n > GP_MAX_SAMPLES:
            logger.warning(
                "GP training on %d samples (> %d). This may be slow. "
                "Consider enabling PCA or switching to RF surrogate.",
                n, GP_MAX_SAMPLES,
            )

        kernel = self._build_kernel()
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X, y)

        optimized_kernel = self._model.kernel_
        logger.info(
            "GP fitted on %d samples | optimized kernel: %s | "
            "log-marginal-likelihood: %.3f",
            n,
            optimized_kernel,
            self._model.log_marginal_likelihood_value_,
        )
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return GP posterior mean and standard deviation.

        σ(x) here is the TRUE posterior standard deviation — not a heuristic.
        It represents the GP's epistemic uncertainty about f(x) given the
        training data, governed by:

            σ²(x*) = k(x*, x*) - k(x*, X)(K + σ²_n I)⁻¹ k(X, x*)

        High σ means the GP has seen few similar training points.
        Low σ means the GP is confident (many nearby labeled samples).
        """
        if self._model is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = self._model.predict(X, return_std=True)

        # Clip std to non-negative (numerical precision can give tiny negatives)
        std = np.maximum(std, 0.0)
        return mean, std

    def predict_with_confidence(self, X: np.ndarray) -> dict:
        """
        Full prediction output including confidence bounds.

        Returns dict with arrays:
            predicted_property, prediction_std,
            lower_confidence_bound, upper_confidence_bound
        """
        mean, std = self.predict(X)
        return {
            "predicted_property": mean,
            "prediction_std": std,
            "lower_confidence_bound": mean - CI_MULTIPLIER * std,
            "upper_confidence_bound": mean + CI_MULTIPLIER * std,
        }

    def get_kernel_params(self) -> dict:
        """Return optimized kernel hyperparameters for logging/display."""
        if self._model is None:
            return {}
        params = self._model.kernel_.get_params()
        return {k: float(v) if hasattr(v, "__float__") else str(v)
                for k, v in params.items()}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def log_marginal_likelihood(self) -> float:
        if self._model is None:
            return float("-inf")
        return float(self._model.log_marginal_likelihood_value_)

    @property
    def n_train(self) -> int:
        return self._n_train


class DeepEnsembleSurrogate(BaseSurrogate):
    """
    Deep Ensemble of MLP regressors (Phase 2 / large-dataset fallback).

    Trains `n_members` independent MLPs with different random seeds.
    Uncertainty = std of the ensemble member predictions.

    Reference: Lakshminarayanan et al. (2017). "Simple and Scalable
    Predictive Uncertainty Estimation using Deep Ensembles."

    This is appropriate when:
    - N_labeled > 1500 (GP becomes too slow)
    - PCA is not desired
    - Richer nonlinear modeling is needed

    Parameters
    ----------
    n_members : int
        Number of ensemble members (5 is standard from the paper).
    hidden_layer_sizes : tuple
        MLP architecture.
    max_iter : int
        Training epochs.
    random_state : int
        Base seed; each member uses seed + member_index.
    """

    def __init__(
        self,
        n_members: int = 5,
        hidden_layer_sizes: tuple = (128, 64, 32),
        max_iter: int = 500,
        learning_rate_init: float = 1e-3,
        random_state: int = 42,
    ):
        self.n_members = n_members
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self._members: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepEnsembleSurrogate":
        from sklearn.neural_network import MLPRegressor

        self._members = []
        for i in range(self.n_members):
            seed = self.random_state + i
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )
            mlp.fit(X, y)
            self._members.append(mlp)

        logger.info(
            "DeepEnsemble fitted: %d members on %d samples",
            self.n_members, X.shape[0],
        )
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._members:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        preds = np.array([m.predict(X) for m in self._members])
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    @property
    def is_fitted(self) -> bool:
        return len(self._members) > 0


def create_bayesian_surrogate(
    model_type: str = "gp",
    random_state: int = 42,
    **kwargs,
) -> BaseSurrogate:
    """Factory for probabilistic surrogate models."""
    registry = {
        "gp": GaussianProcessSurrogate,
        "gaussian_process": GaussianProcessSurrogate,
        "ensemble": DeepEnsembleSurrogate,
        "deep_ensemble": DeepEnsembleSurrogate,
    }
    if model_type.lower() not in registry:
        raise ValueError(
            f"Unknown Bayesian model type '{model_type}'. "
            f"Available: {list(registry.keys())}"
        )
    cls = registry[model_type.lower()]
    return cls(random_state=random_state, **kwargs)
