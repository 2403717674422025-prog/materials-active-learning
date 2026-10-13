"""Pydantic schemas for dataset endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DatasetUploadResponse(BaseModel):
    id: str
    name: str
    status: str
    num_samples: Optional[int] = None
    num_features: Optional[int] = None
    message: str = "Dataset upload initiated"


class DatasetInfo(BaseModel):
    id: str
    name: str
    source: Optional[str] = None
    num_samples: Optional[int] = None
    num_features: Optional[int] = None
    target_column: Optional[str] = None
    composition_column: str = "composition"
    feature_columns: List[str] = []
    global_maximum: Optional[float] = None
    global_minimum: Optional[float] = None
    status: str
    upload_time: datetime
    ready_time: Optional[datetime] = None
    error_message: Optional[str] = None


class DatasetPreview(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int


class DatasetStatistics(BaseModel):
    n_samples: int
    n_features: int
    target_min: float
    target_max: float
    target_mean: float
    target_std: float
    target_median: float
    target_q25: float
    target_q75: float
    target_histogram: Dict[str, Any]
    feature_names: List[str]
    # Bug fix: these were returned by get_feature_statistics but dropped by Pydantic
    feature_means: List[float] = []
    feature_stds: List[float] = []


class LoadBuiltinDatasetRequest(BaseModel):
    dataset_name: str = Field(
        default="matbench_expt_gap",
        description="Name of built-in dataset to load (e.g. matbench_expt_gap)"
    )
    target_column: str = Field(
        default="gap expt",
        description="Name of the target property column"
    )
