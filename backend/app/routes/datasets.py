"""Dataset routes: upload, load built-in, preview, statistics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.schemas.dataset import (
    DatasetInfo,
    DatasetPreview,
    DatasetStatistics,
    DatasetUploadResponse,
    LoadBuiltinDatasetRequest,
)
from backend.app.services.dataset_service import (
    get_dataset_by_id,
    get_dataset_preview,
    get_dataset_statistics,
    list_datasets,
    load_builtin_dataset,
    upload_csv_dataset,
)
from backend.database import get_db

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/", response_model=list[DatasetInfo])
def list_all_datasets(db: Session = Depends(get_db)):
    """List all available datasets."""
    datasets = list_datasets(db)
    return [_to_schema(d) for d in datasets]


@router.post("/load-builtin", response_model=DatasetUploadResponse)
def load_builtin(
    req: LoadBuiltinDatasetRequest,
    db: Session = Depends(get_db),
):
    """
    Load a built-in matminer dataset (e.g. matbench_expt_gap).
    This triggers featurization which may take 1-5 minutes on first load.
    Subsequent loads use the cache and return immediately.
    """
    try:
        record = load_builtin_dataset(db, req.dataset_name, req.target_column)
        return DatasetUploadResponse(
            id=record.id,
            name=record.name,
            status=record.status,
            num_samples=record.num_samples,
            num_features=record.num_features,
            message=f"Dataset '{req.dataset_name}' loaded successfully."
            if record.status == "ready"
            else f"Dataset '{req.dataset_name}' is loading...",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    composition_column: str = Form(default="composition"),
    db: Session = Depends(get_db),
):
    """
    Upload a custom CSV dataset.

    Requirements:
    - Must be a CSV file
    - Must have a composition column (chemical formulas)
    - Must have a numeric target column
    - Maximum file size: 50 MB
    """
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported.",
        )

    content = await file.read()

    try:
        record = upload_csv_dataset(
            db, content, file.filename, target_column, composition_column
        )
        return DatasetUploadResponse(
            id=record.id,
            name=record.name,
            status=record.status,
            num_samples=record.num_samples,
            num_features=record.num_features,
            message="Dataset uploaded and featurized successfully.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/{dataset_id}", response_model=DatasetInfo)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    record = _get_or_404(db, dataset_id)
    return _to_schema(record)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: str,
    n_rows: int = 100,
    db: Session = Depends(get_db),
):
    record = _get_or_404(db, dataset_id)
    _require_ready(record)
    preview = get_dataset_preview(record, n_rows=n_rows)
    return DatasetPreview(**preview)


@router.get("/{dataset_id}/statistics", response_model=DatasetStatistics)
def dataset_statistics(dataset_id: str, db: Session = Depends(get_db)):
    record = _get_or_404(db, dataset_id)
    _require_ready(record)
    stats = get_dataset_statistics(record)
    return DatasetStatistics(**{
        k: stats[k] for k in DatasetStatistics.model_fields
        if k in stats
    })


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_or_404(db, dataset_id):
    record = get_dataset_by_id(db, dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")
    return record


def _require_ready(record):
    if record.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset is not ready (status={record.status}).",
        )


def _to_schema(d) -> DatasetInfo:
    return DatasetInfo(
        id=d.id,
        name=d.name,
        source=d.source,
        num_samples=d.num_samples,
        num_features=d.num_features,
        target_column=d.target_column,
        composition_column=d.composition_column,
        feature_columns=d.feature_columns or [],
        global_maximum=d.global_maximum,
        global_minimum=d.global_minimum,
        status=d.status,
        upload_time=d.upload_time,
        ready_time=d.ready_time,
        error_message=d.error_message,
    )
