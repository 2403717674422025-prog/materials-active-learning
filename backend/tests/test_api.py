"""
Integration tests for FastAPI routes.

Uses a single shared in-memory SQLite connection so all tables are visible
to every session throughout the test run (SQLite :memory: databases are
per-connection; using the same connection avoids "no such table" errors).
"""

import io
import uuid
import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["DATASET_CACHE_DIR"] = "/tmp/test_cache"
os.environ["UPLOAD_DIR"]        = "/tmp/test_uploads"
os.environ["MODEL_DIR"]         = "/tmp/test_models"

from backend.main import app
from backend.database import get_db, Base
from backend.app.models import dataset as ds_model
from backend.app.models.benchmark import BenchmarkRun
from backend.app.models import experiment as exp_model
from backend.app.models import result as res_model
from backend.app.services import tuning_service

# ── Single shared in-memory engine with a persistent connection ───────────────
# Using connect_args={"check_same_thread": False} and a single connection
# keeps all tables visible across sessions within the same test run.
TEST_DB_URL = "sqlite:///:memory:?check_same_thread=False"
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    # Use a single underlying connection for in-memory SQLite
)

# Intercept the "connect" event to reuse the same underlying connection
from sqlalchemy import pool as sa_pool
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=sa_pool.StaticPool,   # <- reuse one connection for all sessions
)

TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Create all tables once using the shared engine
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


# ── Helper ────────────────────────────────────────────────────────────────────
def _create_synthetic_dataset_in_db() -> str:
    """Insert a ready dataset record with a real feature cache into the test DB."""
    rng = np.random.RandomState(0)
    N = 150
    X = pd.DataFrame(rng.randn(N, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.uniform(0, 10, N), name="gap expt")
    candidate_ids = [f"MAT-{i:04d}" for i in range(N)]

    cache_dir = Path("/tmp/test_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = str(uuid.uuid4())
    cache_path = str(cache_dir / f"{dataset_id}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({"X": X, "y": y, "candidate_ids": candidate_ids}, f)

    db = TestSessionLocal()
    record = ds_model.Dataset(
        id=dataset_id, name="test_dataset", source="synthetic",
        num_samples=N, num_features=10, target_column="gap expt",
        composition_column="composition",
        feature_columns=[f"f{i}" for i in range(10)],
        global_maximum=float(y.max()), global_minimum=float(y.min()),
        cache_path=cache_path, status="ready",
    )
    db.add(record)
    db.commit()
    db.close()
    return dataset_id


_PAYLOAD_BASE = {
    "strategy": "random", "model_type": "rf",
    "initial_samples": 20, "batch_size": 5, "total_budget": 50,
    "random_seed": 42, "target_column": "gap expt",
    "xi": 0.01, "kappa": 2.0,
    "alpha": 0.4, "beta": 0.35, "gamma": 0.25,
    "use_pca": False, "pca_n_components": 50,
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestDatasetEndpoints:
    def test_list_datasets_empty(self):
        r = client.get("/datasets/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_upload_non_csv_returns_400(self):
        r = client.post(
            "/datasets/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"target_column": "target"},
        )
        assert r.status_code == 400

    def test_upload_missing_target_column_returns_422(self):
        csv_content = b"composition,some_col\nFe2O3,1.2\nTiO2,2.3\n"
        r = client.post(
            "/datasets/upload",
            files={"file": ("test.csv", csv_content, "text/csv")},
            data={"target_column": "nonexistent_col"},
        )
        assert r.status_code == 422

    def test_get_nonexistent_dataset_returns_404(self):
        r = client.get("/datasets/nonexistent-id-xyz")
        assert r.status_code == 404


class TestExperimentEndpoints:
    def setup_method(self):
        self.dataset_id = _create_synthetic_dataset_in_db()

    def _create(self, **overrides) -> str:
        payload = {**_PAYLOAD_BASE, "dataset_id": self.dataset_id, **overrides}
        r = client.post("/experiments/", json=payload)
        assert r.status_code == 201, r.text
        return r.json()["id"]

    def _init(self) -> str:
        exp_id = self._create()
        r = client.post(f"/experiments/{exp_id}/initialize")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "initialized"
        return exp_id

    def test_create_experiment_missing_dataset(self):
        payload = {**_PAYLOAD_BASE, "dataset_id": "nonexistent-ds"}
        r = client.post("/experiments/", json=payload)
        assert r.status_code == 422

    def test_create_experiment_success(self):
        exp_id = self._create()
        assert len(exp_id) > 8

    def test_initialize_experiment(self):
        exp_id = self._init()
        assert exp_id is not None

    def test_run_iteration(self):
        exp_id = self._init()
        r = client.post(f"/experiments/{exp_id}/run-iteration")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["iteration"] == 1
        assert len(data["selected_candidate_ids"]) == 5
        assert data["test_rmse"] >= 0

    def test_get_status(self):
        exp_id = self._init()
        r = client.get(f"/experiments/{exp_id}/status")
        assert r.status_code == 200
        data = r.json()
        assert "budget_used" in data
        assert data["budget_total"] == 50

    def test_get_history_after_iteration(self):
        exp_id = self._init()
        client.post(f"/experiments/{exp_id}/run-iteration")
        r = client.get(f"/experiments/{exp_id}/history")
        assert r.status_code == 200
        history = r.json()
        assert len(history) == 1
        assert history[0]["iteration"] == 1

    def test_get_candidates(self):
        exp_id = self._init()
        client.post(f"/experiments/{exp_id}/run-iteration")
        r = client.get(f"/experiments/{exp_id}/candidates?page=1&page_size=20")
        assert r.status_code == 200
        data = r.json()
        assert "candidates" in data
        assert "total_candidates" in data

    def test_calibration_endpoint_reports_insufficient_pairs(self):
        exp_id = self._init()
        response = client.get(f"/experiments/{exp_id}/calibration")
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["n_aligned"] == 0
        assert data["minimum_required"] == 5

    def test_get_persisted_benchmark(self):
        exp_id = self._create()
        db = TestSessionLocal()
        record = BenchmarkRun(
            experiment_id=exp_id,
            status="completed",
            model_type="rf",
            configuration={"protocol_version": "matched-multiseed-v1"},
            results=[{"strategy": "random", "seed": 42}],
            summary={"best_strategy": "random"},
        )
        db.add(record)
        db.commit()
        benchmark_id = record.id
        db.close()

        response = client.get(
            f"/experiments/{exp_id}/benchmarks/{benchmark_id}"
        )
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["results"][0]["seed"] == 42

        missing = client.get(
            f"/experiments/{exp_id}/benchmarks/not-a-benchmark"
        )
        assert missing.status_code == 404

    def test_list_experiments_endpoint(self):
        """New endpoint: GET /experiments/ returns a list."""
        self._create()
        r = client.get("/experiments/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_budget_constraint_invalid(self):
        payload = {**_PAYLOAD_BASE, "dataset_id": self.dataset_id,
                   "initial_samples": 100, "total_budget": 50}
        r = client.post("/experiments/", json=payload)
        assert r.status_code == 422

    def test_tuning_request_validation(self):
        exp_id = self._init()
        unsupported = client.post(
            f"/experiments/{exp_id}/tuning-runs",
            json={"model_type": "ensemble", "n_trials": 5, "random_seed": 42},
        )
        too_many_trials = client.post(
            f"/experiments/{exp_id}/tuning-runs",
            json={"model_type": "rf", "n_trials": 101, "random_seed": 42},
        )
        assert unsupported.status_code == 422
        assert too_many_trials.status_code == 422

    def test_tuning_run_status_and_apply(self, monkeypatch):
        exp_id = self._init()
        monkeypatch.setattr(tuning_service, "SessionLocal", TestSessionLocal)
        response = client.post(
            f"/experiments/{exp_id}/tuning-runs",
            json={
                "model_type": "rf",
                "n_trials": 1,
                "random_seed": 42,
                "timeout_seconds": 60,
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]

        status = client.get(f"/experiments/{exp_id}/tuning-runs/{run_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "completed"
        assert status.json()["best_validation_rmse"] is not None

        applied = client.post(f"/experiments/{exp_id}/tuning-runs/{run_id}/apply")
        assert applied.status_code == 200, applied.text
        assert applied.json()["applied_to_experiment"] is True
        iteration = client.post(f"/experiments/{exp_id}/run-iteration")
        assert iteration.status_code == 200, iteration.text
        assert iteration.json()["iteration"] == 1
