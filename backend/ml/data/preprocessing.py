"""
Preprocessing pipeline: scaling, splitting, and state management.

This module handles:
 - Train/test splitting with reproducible seeds
 - Variance/correlation feature filtering fit ONLY on training data
 - StandardScaler that is fit ONLY on the current labeled set
 - Optional PCA for GP scalability
 - Serialization/deserialization of preprocessing state
   so inference always uses exactly the same transformations

Design invariant:
    The scaler is NEVER fit on the unlabeled pool or the test set.
    Fitting happens only on the labeled training set at each AL iteration.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""

    test_size: float = 0.20          # fraction of total data held as test
    random_state: int = 42
    use_pca: bool = False            # activate PCA for GP scalability
    pca_n_components: int = 50       # PCA dims when use_pca=True
    pca_variance_threshold: float = 0.95  # alt: keep this fraction of variance
    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.95

    # Auto-switch to PCA when labeled set grows beyond this size
    # (GP cubic complexity kicks in around N=800)
    pca_auto_threshold: int = 800


@dataclass
class FeatureFilterState:
    """Training-fitted imputation and feature selection rules."""

    input_columns: list[str] = field(default_factory=list)
    kept_columns: list[str] = field(default_factory=list)
    imputation_values: dict[str, float] = field(default_factory=dict)
    fit_sample_count: int = 0
    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.95

    @classmethod
    def fit(
        cls,
        X_train: pd.DataFrame,
        variance_threshold: float = 1e-6,
        correlation_threshold: float = 0.95,
    ) -> "FeatureFilterState":
        if X_train.empty:
            raise ValueError("Cannot fit feature filters on an empty training set.")

        input_columns = list(X_train.columns)
        numeric = X_train.reindex(columns=input_columns).apply(
            pd.to_numeric, errors="coerce"
        )
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        medians = numeric.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        imputed = numeric.fillna(medians)

        variances = imputed.var(axis=0)
        variance_columns = list(variances[variances > variance_threshold].index)
        if not variance_columns:
            raise ValueError(
                "No varying numeric features remain after training-only variance filtering."
            )

        kept_columns = variance_columns
        if len(variance_columns) > 1:
            corr = imputed[variance_columns].corr().abs()
            upper = corr.where(
                np.triu(np.ones(corr.shape), k=1).astype(bool)
            )
            correlated = [
                column
                for column in upper.columns
                if any(upper[column] > correlation_threshold)
            ]
            kept_columns = [
                column for column in variance_columns if column not in correlated
            ]

        if not kept_columns:
            raise ValueError(
                "No features remain after training-only correlation filtering."
            )

        return cls(
            input_columns=input_columns,
            kept_columns=kept_columns,
            imputation_values={column: float(medians[column]) for column in input_columns},
            fit_sample_count=len(X_train),
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align to the training schema, impute with training medians, and filter."""
        aligned = X.reindex(columns=self.input_columns).apply(
            pd.to_numeric, errors="coerce"
        )
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        aligned = aligned.fillna(self.imputation_values)
        return aligned.reindex(columns=self.kept_columns)


@dataclass
class PreprocessingState:
    """
    Carries the fitted preprocessing objects for one AL iteration.

    Serialized after each fit() call so inference replicates exact transforms.
    """

    scaler: Optional[StandardScaler] = None
    pca: Optional[PCA] = None
    feature_columns: list = field(default_factory=list)
    feature_filter: Optional[FeatureFilterState] = None
    config: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted feature selection, scaler, and optional PCA."""
        feature_filter = getattr(self, "feature_filter", None)
        if feature_filter is not None:
            X_selected = feature_filter.transform(X)
        else:
            # Compatibility with preprocessing states serialized before feature
            # filtering was added.
            X_selected = X.reindex(columns=self.feature_columns).apply(
                pd.to_numeric, errors="coerce"
            )
            X_selected = X_selected.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_aligned = X_selected[self.feature_columns].values.astype(float)
        if self.scaler is not None:
            X_aligned = self.scaler.transform(X_aligned)
        if self.pca is not None:
            X_aligned = self.pca.transform(X_aligned)
        return X_aligned

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("PreprocessingState saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "PreprocessingState":
        with open(path, "rb") as f:
            state = pickle.load(f)
        logger.info("PreprocessingState loaded from %s", path)
        return state


def initial_split(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_ids: list,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    Split the full dataset into a fixed held-out test set and the active
    learning pool. The test set is NEVER touched during the AL loop.

    Returns
    -------
    dict with keys:
        pool_X, pool_y, pool_ids  — candidates available for the AL loop
        test_X,  test_y,  test_ids — fixed evaluation set (never labeled)
    """
    idx = np.arange(len(X))
    pool_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=True
    )

    pool_X = X.iloc[pool_idx].reset_index(drop=True)
    pool_y = y.iloc[pool_idx].reset_index(drop=True)
    pool_ids = [candidate_ids[i] for i in pool_idx]

    test_X = X.iloc[test_idx].reset_index(drop=True)
    test_y = y.iloc[test_idx].reset_index(drop=True)
    test_ids = [candidate_ids[i] for i in test_idx]

    logger.info(
        "Dataset split → pool: %d samples, test: %d samples",
        len(pool_X),
        len(test_X),
    )
    return {
        "pool_X": pool_X,
        "pool_y": pool_y,
        "pool_ids": pool_ids,
        "test_X": test_X,
        "test_y": test_y,
        "test_ids": test_ids,
    }


def seed_labeled_set(
    pool_X: pd.DataFrame,
    pool_y: pd.Series,
    pool_ids: list,
    n_initial: int = 100,
    random_state: int = 42,
    strategy: str = "random",
) -> Tuple[dict, dict]:
    """
    Select the initial labeled seed from the pool.

    Parameters
    ----------
    strategy : str
        'random' — uniform random selection
        'diverse' — select most spread-out samples (max-min distance seeding)

    Returns
    -------
    labeled : dict with labeled_X, labeled_y, labeled_ids
    unlabeled : dict with unlabeled_X, unlabeled_y, unlabeled_ids
    """
    rng = np.random.RandomState(random_state)
    n_pool = len(pool_X)
    n_initial = min(n_initial, n_pool - 1)

    if strategy == "random":
        seed_idx = rng.choice(n_pool, size=n_initial, replace=False)
    elif strategy == "diverse":
        # Simple greedy farthest-point sampling for better initial coverage
        from sklearn.metrics.pairwise import euclidean_distances
        X_arr = pool_X.values
        first = rng.randint(n_pool)
        selected = [first]
        remaining = list(range(n_pool))
        remaining.remove(first)
        while len(selected) < n_initial:
            dists = euclidean_distances(
                X_arr[remaining], X_arr[selected]
            ).min(axis=1)
            next_idx = remaining[int(np.argmax(dists))]
            selected.append(next_idx)
            remaining.remove(next_idx)
        seed_idx = np.array(selected)
    else:
        raise ValueError(f"Unknown seed strategy: {strategy}")

    all_idx = np.arange(n_pool)
    unlabeled_idx = np.array([i for i in all_idx if i not in seed_idx])

    labeled = {
        "labeled_X": pool_X.iloc[seed_idx].reset_index(drop=True),
        "labeled_y": pool_y.iloc[seed_idx].reset_index(drop=True),
        "labeled_ids": [pool_ids[i] for i in seed_idx],
    }
    unlabeled = {
        "unlabeled_X": pool_X.iloc[unlabeled_idx].reset_index(drop=True),
        "unlabeled_y": pool_y.iloc[unlabeled_idx].reset_index(drop=True),
        "unlabeled_ids": [pool_ids[i] for i in unlabeled_idx],
    }

    logger.info(
        "Seed: %d labeled | %d unlabeled remaining",
        len(labeled["labeled_X"]),
        len(unlabeled["unlabeled_X"]),
    )
    return labeled, unlabeled


def fit_preprocessor(
    labeled_X: pd.DataFrame,
    config: PreprocessingConfig,
    labeled_set_size: Optional[int] = None,
) -> PreprocessingState:
    """
    Fit feature filters, StandardScaler, and optionally PCA on training data.

    CRITICAL: In active learning this must be called ONLY on the currently
    labeled set at the start of each iteration. The fitted state is then
    applied to unlabeled candidates, validation data, and test data without
    refitting any feature-selection or transformation rules.

    Parameters
    ----------
    labeled_X : pd.DataFrame
        Current labeled feature matrix.
    config : PreprocessingConfig
    labeled_set_size : int, optional
        Override for PCA auto-switch threshold check.
    """
    feature_filter = FeatureFilterState.fit(
        labeled_X,
        variance_threshold=config.variance_threshold,
        correlation_threshold=config.correlation_threshold,
    )
    X_selected = feature_filter.transform(labeled_X)
    state = PreprocessingState(config=config, feature_filter=feature_filter)
    state.feature_columns = list(feature_filter.kept_columns)
    X_arr = X_selected.values.astype(float)

    # Fit StandardScaler on labeled set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    state.scaler = scaler

    # Determine whether to apply PCA
    n_samples = labeled_set_size or len(labeled_X)
    use_pca = config.use_pca or (n_samples >= config.pca_auto_threshold)

    if use_pca and X_scaled.shape[1] > config.pca_n_components:
        n_components = min(config.pca_n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=config.random_state)
        pca.fit(X_scaled)
        state.pca = pca
        explained = pca.explained_variance_ratio_.sum()
        logger.info(
            "PCA fitted: %d → %d dims, explained variance: %.1f%%",
            X_scaled.shape[1],
            n_components,
            explained * 100,
        )
    else:
        state.pca = None

    return state


def apply_preprocessor(
    state: PreprocessingState,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Apply a fitted PreprocessingState to transform features.
    Safe to call on labeled set, unlabeled pool, or test set.
    """
    return state.transform(X)


def get_feature_statistics(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute descriptive statistics for the UI Dataset Explorer page.
    """
    stats = {
        "n_samples": int(len(X)),
        "n_features": int(len(X.columns)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "target_median": float(y.median()),
        "target_q25": float(y.quantile(0.25)),
        "target_q75": float(y.quantile(0.75)),
        "feature_names": list(X.columns),
        "feature_means": X.mean().tolist(),
        "feature_stds": X.std().tolist(),
    }

    # Target distribution histogram (20 bins)
    hist, bin_edges = np.histogram(y.values, bins=20)
    stats["target_histogram"] = {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
    }

    return stats
