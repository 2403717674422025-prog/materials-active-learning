"""
Feature engineering pipeline for materials composition data.

Converts raw chemical composition strings into numerical feature vectors
using matminer featurizers based on the Magpie descriptor set.

Pipeline:
    Raw composition string
        → pymatgen Composition object
        → ElementProperty (Magpie) — 132 features
        → Stoichiometry            —   5 features
        → ValenceOrbital           —   4 features
        → IonProperty              —   2 features
        → BandCenter               —   1 feature
        → Preserve all numerical descriptors in the feature cache
        → Fit near-constant/correlation filters on each training partition
        → Model-ready feature matrix
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
FEATURE_CACHE_VERSION = 2

# Suppress verbose matminer/pymatgen warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _check_imports() -> bool:
    """Verify that matminer and pymatgen are available."""
    try:
        import matminer  # noqa: F401
        import pymatgen  # noqa: F401
        return True
    except ImportError:
        return False


def composition_to_features(
    df: pd.DataFrame,
    composition_col: str = "composition",
    drop_low_variance: bool = True,
    variance_threshold: float = 1e-6,
    drop_correlated: bool = True,
    correlation_threshold: float = 0.95,
    feature_columns: Optional[list] = None,
) -> Tuple[pd.DataFrame, list, list]:
    """
    Convert composition strings to numerical feature vectors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing composition strings.
    composition_col : str
        Name of the column with composition strings.
    drop_low_variance : bool
        Whether to drop near-constant features.
    variance_threshold : float
        Minimum variance to keep a feature.
    drop_correlated : bool
        Whether to drop highly correlated feature pairs.
    correlation_threshold : float
        Maximum absolute Pearson correlation to tolerate.
    feature_columns : list, optional
        If provided, use only these columns (for inference-time consistency).

    Returns
    -------
    features_df : pd.DataFrame
        Numerical feature matrix (rows = samples, columns = features).
    kept_columns : list
        Names of feature columns that survived filtering.
    dropped_indices : list
        Integer indices of rows that failed featurization and were dropped.
    """
    if not _check_imports():
        raise ImportError(
            "matminer and pymatgen are required for feature engineering. "
            "Install with: pip install matminer pymatgen"
        )

    from matminer.featurizers.composition import (
        BandCenter,
        ElementProperty,
        IonProperty,
        Stoichiometry,
        ValenceOrbital,
    )
    from pymatgen.core import Composition

    logger.info("Starting featurization of %d compositions", len(df))

    # ── Step 1: parse composition strings ────────────────────────────────────
    df = df.copy()
    compositions = []
    failed_indices = []
    for idx, row in df.iterrows():
        try:
            comp = Composition(str(row[composition_col]))
            compositions.append(comp)
        except Exception as exc:
            logger.warning("Failed to parse composition at index %s: %s", idx, exc)
            compositions.append(None)
            failed_indices.append(idx)

    df["_comp"] = compositions
    n_failed = len(failed_indices)
    if n_failed > 0:
        logger.warning("Dropping %d rows with unparseable compositions", n_failed)
    df_valid = df[df["_comp"].notna()].copy()

    # ── Step 2: apply featurizers ─────────────────────────────────────────────
    featurizer_list = [
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
        ValenceOrbital(),
        IonProperty(fast=True),
        BandCenter(),
    ]

    # Drop the original string column before renaming the parsed Composition
    # objects; otherwise pandas keeps duplicate composition labels and the
    # matminer featurizers receive a two-column DataFrame instead of a Series.
    df_valid = df_valid.drop(columns=[composition_col]).rename(columns={"_comp": composition_col})
    successful_rows = list(df_valid.index)
    failed_featurize = []

    for feat in featurizer_list:
        feat.set_n_jobs(1)  # avoid multiprocessing issues in server context
        try:
            feat.fit(df_valid[composition_col])
            df_valid = feat.featurize_dataframe(
                df_valid,
                col_id=composition_col,
                ignore_errors=True,
                inplace=False,
            )
            logger.info("Applied featurizer: %s", feat.__class__.__name__)
        except Exception as exc:
            logger.error("Featurizer %s failed: %s", feat.__class__.__name__, exc)

    # ── Step 3: isolate feature columns ──────────────────────────────────────
    non_feature_cols = [c for c in df.columns if c != "_comp"] + [composition_col]
    feature_cols = [c for c in df_valid.columns if c not in non_feature_cols]
    features_df = df_valid[feature_cols].copy()

    # Drop rows that still have NaN after featurization
    nan_mask = features_df.isna().any(axis=1)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        logger.warning("Dropping %d rows with NaN features after featurization", n_nan)
        failed_featurize.extend(list(features_df[nan_mask].index))
        features_df = features_df[~nan_mask]

    all_dropped = list(set(failed_indices + failed_featurize))

    # Convert all to float
    features_df = features_df.astype(float)

    # ── Step 4: if feature_columns provided, align to training-time schema ────
    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in features_df.columns]
        if missing:
            logger.warning(
                "Inference: %d feature columns missing, filling with 0", len(missing)
            )
            for m in missing:
                features_df[m] = 0.0
        features_df = features_df[feature_columns]
        return features_df, feature_columns, all_dropped

    # ── Step 5: drop near-constant features ───────────────────────────────────
    if drop_low_variance:
        variances = features_df.var(axis=0)
        keep_mask = variances > variance_threshold
        n_dropped_var = (~keep_mask).sum()
        if n_dropped_var > 0:
            logger.info("Dropping %d near-constant features", n_dropped_var)
        features_df = features_df.loc[:, keep_mask]

    # ── Step 6: drop highly correlated features ───────────────────────────────
    if drop_correlated and len(features_df.columns) > 1:
        corr_matrix = features_df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col
            for col in upper_tri.columns
            if any(upper_tri[col] > correlation_threshold)
        ]
        if to_drop:
            logger.info(
                "Dropping %d highly correlated features (threshold=%.2f)",
                len(to_drop),
                correlation_threshold,
            )
            features_df = features_df.drop(columns=to_drop)

    kept_columns = list(features_df.columns)
    logger.info(
        "Featurization complete: %d samples, %d features",
        len(features_df),
        len(kept_columns),
    )

    return features_df, kept_columns, all_dropped


def load_and_featurize_dataset(
    csv_path: Optional[str] = None,
    composition_col: str = "composition",
    target_col: str = "gap expt",
    cache_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Load a materials dataset, featurize it, and optionally cache the result.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file. If None, downloads matbench_expt_gap automatically.
    composition_col : str
        Column name containing composition strings.
    target_col : str
        Column name containing the target property.
    cache_path : str, optional
        Path to save/load featurized data as a pickle for faster re-loading.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values.
    candidate_ids : list
        String identifiers for each sample (composition strings).
    """
    import pickle

    # ── Try cache first ───────────────────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            cached_X, cached_y, cached_ids = cached["X"], cached["y"], cached["candidate_ids"]
            if (
                cached.get("feature_cache_version") == FEATURE_CACHE_VERSION
                and len(cached_X) > 0
                and len(cached_X.columns) > 0
                and len(cached_y) == len(cached_X) == len(cached_ids)
            ):
                logger.info("Loading featurized data from cache: %s", cache_path)
                return cached_X, cached_y, cached_ids
            logger.warning(
                "Ignoring stale, empty, or inconsistent dataset cache: %s", cache_path
            )
        except (OSError, EOFError, TypeError, ValueError, KeyError, pickle.PickleError, AttributeError) as exc:
            logger.warning("Ignoring unreadable dataset cache %s: %s", cache_path, exc)

    # ── Load raw data ─────────────────────────────────────────────────────────
    if csv_path and Path(csv_path).exists():
        logger.info("Loading dataset from: %s", csv_path)
        df_raw = pd.read_csv(csv_path)
        # Handle matminer's serialized composition column
        if composition_col not in df_raw.columns and "composition" in df_raw.columns:
            composition_col = "composition"
        if composition_col not in df_raw.columns:
            raise ValueError(
                f"Composition column '{composition_col}' not found. "
                f"Available columns: {list(df_raw.columns)}"
            )
    else:
        logger.info("Downloading matbench_expt_gap from matminer...")
        try:
            from matminer.datasets import load_dataset
            df_raw = load_dataset("matbench_expt_gap")
            # matminer returns pymatgen Composition objects; convert to strings
            if hasattr(df_raw[composition_col].iloc[0], "formula"):
                df_raw[composition_col] = df_raw[composition_col].apply(
                    lambda c: c.formula if hasattr(c, "formula") else str(c)
                )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load dataset. Ensure matminer is installed "
                f"and internet is available. Error: {exc}"
            )

    if target_col not in df_raw.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df_raw.columns)}"
        )

    # Record candidate IDs (composition strings) BEFORE featurization
    candidate_ids_raw = df_raw[composition_col].astype(str).tolist()

    # ── Featurize ─────────────────────────────────────────────────────────────
    # Feature filters are fitted later on each training partition by
    # fit_preprocessor(); never select columns using the complete dataset here.
    features_df, _, dropped_indices = composition_to_features(
        df_raw, drop_low_variance=False, drop_correlated=False
    )

    # Align target and candidate IDs to surviving rows
    surviving_mask = ~df_raw.index.isin(dropped_indices)
    y = df_raw.loc[surviving_mask, target_col].reset_index(drop=True)
    y.name = target_col
    candidate_ids = [
        str(candidate_ids_raw[i])
        for i, keep in enumerate(surviving_mask)
        if keep
    ]
    X = features_df.reset_index(drop=True)

    logger.info(
        "Final dataset: %d samples, %d features, target='%s'",
        len(X),
        len(X.columns),
        target_col,
    )

    # ── Cache result ──────────────────────────────────────────────────────────
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            import pickle
            pickle.dump(
                {
                    "X": X,
                    "y": y,
                    "candidate_ids": candidate_ids,
                    "feature_cache_version": FEATURE_CACHE_VERSION,
                },
                f,
            )
        logger.info("Featurized data cached to: %s", cache_path)

    return X, y, candidate_ids
