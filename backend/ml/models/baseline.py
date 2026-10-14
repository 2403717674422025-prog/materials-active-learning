"""
Baseline predictive models.

Implements:
  - RandomForestSurrogate  — primary non-Bayesian baseline
  - XGBoostSurrogate       — secondary comparison model

Both expose a unified interface:
    fit(X, y)
    predict(X)  → (mean, std)   std = ensemble variance-based uncertainty

The variance-based uncertainty from RF (std of individual tree predictions)
is not a true Bayesian posterior but is a well-validated proxy widely used
in materials informatics (see: Breiman 2001, Geurts et al. 2006).
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class BaseSurrogate(ABC):
    """Abstract interface that all surrogate models must implement."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSurrogate":
        """Train the model on labeled data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (mean, std) predictions for every row in X.

        Returns
        -------
        mean : np.ndarray, shape (n,)
        std  : np.ndarray, shape (n,)   predictive uncertainty
        """

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute standard regression metrics on a held-out set."""
        mean, _ = self.predict(X)
        mae = mean_absolute_error(y, mean)
        rmse = float(np.sqrt(mean_squared_error(y, mean)))
        r2 = r2_score(y, mean)
        return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("%s saved to %s", self.__class__.__name__, path)

    @classmethod
    def load(cls, path: str) -> "BaseSurrogate":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Surrogate loaded from %s", path)
        return model


class RandomForestSurrogate(BaseSurrogate):
    """
    Random Forest surrogate with variance-based uncertainty.

    Uncertainty estimate:
        σ_RF(x) = std({ tree_i(x) : i = 1..n_estimators })

    This is computed from the individual tree predictions stored in
    rf.estimators_. It is NOT a Bayesian posterior standard deviation,
    but correlates well with actual prediction error and is sufficient
    for uncertainty sampling and hybrid acquisition functions.

    Parameters
    ----------
    n_estimators : int
        Number of trees (200 gives stable variance estimates).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        max_features: str | float | None = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model: RandomForestRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestSurrogate":
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X, y)
        logger.info(
            "RandomForestSurrogate fitted on %d samples, %d features",
            X.shape[0], X.shape[1],
        )
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # Collect predictions from each tree: shape (n_estimators, n_samples)
        tree_preds = np.array(
            [tree.predict(X) for tree in self._model.estimators_]
        )
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0)
        return mean, std

    @property
    def is_fitted(self) -> bool:
        return self._model is not None


class XGBoostSurrogate(BaseSurrogate):
    """
    XGBoost surrogate with bootstrap-ensemble uncertainty.

    Trains `n_bootstrap` independent XGBoost models on bootstrap samples
    of the training data. Uncertainty = std of the ensemble predictions.

    This provides a stronger point-prediction baseline than RF but uses
    the same ensemble-variance uncertainty approach.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_bootstrap: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self._models: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostSurrogate":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        self._models = []
        rng = np.random.RandomState(self.random_state)
        n = len(X)

        for i in range(self.n_bootstrap):
            seed = int(rng.randint(0, 99999))
            bootstrap_idx = rng.choice(n, size=n, replace=True)
            X_boot, y_boot = X[bootstrap_idx], y[bootstrap_idx]

            model = XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=seed,
                verbosity=0,
                tree_method="hist",
            )
            model.fit(X_boot, y_boot)
            self._models.append(model)

        logger.info(
            "XGBoostSurrogate fitted: %d bootstrap models on %d samples",
            self.n_bootstrap, X.shape[0],
        )
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._models:
            raise RuntimeError("Model not fitted. Call fit() first.")
        preds = np.array([m.predict(X) for m in self._models])
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    @property
    def is_fitted(self) -> bool:
        return len(self._models) > 0


def create_surrogate(model_type: str, random_state: int = 42, **kwargs) -> BaseSurrogate:
    """Factory function to create surrogate models by name."""
    models = {
        "rf": RandomForestSurrogate,
        "random_forest": RandomForestSurrogate,
        "xgb": XGBoostSurrogate,
        "xgboost": XGBoostSurrogate,
    }
    if model_type.lower() not in models:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(models.keys())}"
        )
    cls = models[model_type.lower()]
    return cls(random_state=random_state, **kwargs)
