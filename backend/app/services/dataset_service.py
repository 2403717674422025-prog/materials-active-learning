"""
Dataset service: loading, featurizing, caching, and serving dataset info.
"""

from __future__ import annotations

import logging
import os
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.models.dataset import Dataset
from backend.config import get_settings
from backend.ml.data.feature_engineering import FEATURE_CACHE_VERSION
from backend.ml.data.preprocessing import get_feature_statistics

logger = logging.getLogger(__name__)
settings = get_settings()


def load_builtin_dataset(
    db: Session,
    dataset_name: str = "matbench_expt_gap",
    target_column: str = "gap expt",
) -> Dataset:
    """
    Load a built-in matminer dataset, featurize it, and persist to DB.
    Returns the Dataset ORM record.
    """
    from backend.ml.data.feature_engineering import load_and_featurize_dataset

    # Keep older dataset records intact: their globally filtered feature matrix
    # may still be referenced by existing experiments.
    dataset_records = db.query(Dataset).filter(Dataset.source == dataset_name).all()
    dataset_record = None
    for candidate in dataset_records:
        if candidate.status != "ready":
            continue
        try:
            with open(candidate.cache_path, "rb") as cache_file:
                cached = pickle.load(cache_file)
            cached_X = cached.get("X")
            cached_y = cached.get("y")
            cache_is_current = bool(
                cached.get("feature_cache_version") == FEATURE_CACHE_VERSION
                and cached_X is not None and cached_y is not None
                and len(cached_X) > 0 and len(cached_X.columns) > 0
                and len(cached_y) == len(cached_X)
            )
        except (OSError, EOFError, TypeError, ValueError, pickle.PickleError, AttributeError):
            cache_is_current = False
        if (
            cache_is_current
            and (candidate.num_samples or 0) > 0
            and (candidate.num_features or 0) > 0
        ):
            dataset_record = candidate
            logger.info(
                "Dataset '%s' already loaded with current feature schema (%d rows)",
                dataset_name, candidate.num_samples,
            )
            return dataset_record

    dataset_record = Dataset(
        id=str(uuid.uuid4()), name=dataset_name, source=dataset_name,
        target_column=target_column, status="pending", upload_time=datetime.utcnow(),
        preprocessing_config={"feature_cache_version": FEATURE_CACHE_VERSION},
    )
    db.add(dataset_record)
    db.commit()

    try:
        cache_dir = Path(settings.dataset_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / f"{dataset_name}_{dataset_record.id[:8]}.pkl")

        logger.info("Featurizing dataset '%s'...", dataset_name)
        X, y, candidate_ids = load_and_featurize_dataset(
            csv_path=None,
            target_col=target_column,
            cache_path=cache_path,
        )

        stats = get_feature_statistics(X, y)

        dataset_record.num_samples = int(len(X))
        dataset_record.num_features = int(len(X.columns))
        dataset_record.feature_columns = list(X.columns)
        dataset_record.global_maximum = float(y.max())
        dataset_record.global_minimum = float(y.min())
        dataset_record.cache_path = cache_path
        dataset_record.status = "ready"
        dataset_record.ready_time = datetime.utcnow()
        db.commit()

        logger.info(
            "Dataset '%s' ready: %d samples, %d features",
            dataset_name, len(X), len(X.columns),
        )
    except Exception as exc:
        logger.error("Dataset loading failed: %s", exc)
        dataset_record.status = "error"
        dataset_record.error_message = str(exc)
        db.commit()
        raise

    return dataset_record


def get_dataset_by_id(db: Session, dataset_id: str) -> Optional[Dataset]:
    return db.query(Dataset).filter(Dataset.id == dataset_id).first()


def list_datasets(db: Session) -> list:
    return db.query(Dataset).all()


def load_dataset_arrays(dataset_record: Dataset):
    """Load featurized X, y, candidate_ids from cache."""
    if not dataset_record.cache_path or not Path(dataset_record.cache_path).exists():
        raise FileNotFoundError(
            f"Dataset cache not found at {dataset_record.cache_path}. "
            "Re-load the dataset."
        )
    with open(dataset_record.cache_path, "rb") as f:
        cached = pickle.load(f)
    return cached["X"], cached["y"], cached["candidate_ids"]


def get_dataset_preview(dataset_record: Dataset, n_rows: int = 100) -> dict:
    """Return the first n_rows of raw data for the Dataset Explorer page."""
    X, y, ids = load_dataset_arrays(dataset_record)
    preview_df = X.head(n_rows).copy()
    preview_df["composition"] = ids[:n_rows]
    preview_df[dataset_record.target_column] = y.values[:n_rows]
    return {
        "columns": list(preview_df.columns),
        "rows": preview_df.to_dict(orient="records"),
        "total_rows": len(X),
    }


def get_dataset_statistics(dataset_record: Dataset) -> dict:
    """Return full feature + target statistics for Dataset Explorer."""
    X, y, _ = load_dataset_arrays(dataset_record)
    return get_feature_statistics(X, y)


def upload_csv_dataset(
    db: Session,
    file_content: bytes,
    filename: str,
    target_column: str,
    composition_column: str = "composition",
) -> Dataset:
    """
    Accept a user-uploaded CSV, validate it, featurize it, and store.
    """
    import io
    from backend.ml.data.feature_engineering import composition_to_features

    # Validate file size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_content) > max_bytes:
        raise ValueError(
            f"File too large ({len(file_content)//1024//1024}MB). "
            f"Maximum allowed: {settings.max_upload_size_mb}MB"
        )

    # Parse CSV
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}")

    # Validate columns
    if composition_column not in df.columns:
        raise ValueError(
            f"Composition column '{composition_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate target is numeric
    try:
        df[target_column] = pd.to_numeric(df[target_column])
    except Exception:
        raise ValueError(f"Target column '{target_column}' must be numeric.")

    # Save raw CSV
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = str(uuid.uuid4())
    raw_path = str(upload_dir / f"{dataset_id}_{filename}")
    with open(raw_path, "wb") as f:
        f.write(file_content)

    # Create DB record
    record = Dataset(
        id=dataset_id,
        name=filename,
        filename=raw_path,
        target_column=target_column,
        composition_column=composition_column,
        status="pending",
        upload_time=datetime.utcnow(),
        preprocessing_config={"feature_cache_version": FEATURE_CACHE_VERSION},
    )
    db.add(record)
    db.commit()

    try:
        # Featurize
        from backend.ml.data.feature_engineering import load_and_featurize_dataset
        cache_dir = Path(settings.dataset_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / f"upload_{dataset_id[:8]}.pkl")

        X, y, cids = load_and_featurize_dataset(
            csv_path=raw_path,
            composition_col=composition_column,
            target_col=target_column,
            cache_path=cache_path,
        )

        record.num_samples = int(len(X))
        record.num_features = int(len(X.columns))
        record.feature_columns = list(X.columns)
        record.global_maximum = float(y.max())
        record.global_minimum = float(y.min())
        record.cache_path = cache_path
        record.status = "ready"
        record.ready_time = datetime.utcnow()
        db.commit()
    except Exception as exc:
        record.status = "error"
        record.error_message = str(exc)
        db.commit()
        raise

    return record
