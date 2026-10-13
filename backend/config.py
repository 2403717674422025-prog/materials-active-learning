"""
Application configuration loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
Never hardcode secrets — use .env file or environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Uncertainty-Guided Bayesian Active Learning for Materials Discovery"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./materials_al.db"
    # For PostgreSQL: postgresql+asyncpg://user:password@host:5432/dbname

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: List[str] = [
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alternative dev port
        "http://localhost:80",
        "http://frontend:5173",    # Docker service name
    ]

    # ── File Storage ─────────────────────────────────────────────────────────
    upload_dir: str = "./uploads"
    model_dir: str = "./models"
    data_dir: str = "./data"
    max_upload_size_mb: int = 50

    # ── ML Defaults ──────────────────────────────────────────────────────────
    default_model_type: str = "gp"
    default_strategy: str = "puda"
    default_initial_samples: int = 100
    default_batch_size: int = 20
    default_total_budget: int = 300
    default_random_seed: int = 42

    # ── Dataset ──────────────────────────────────────────────────────────────
    default_dataset: str = "matbench_expt_gap"
    dataset_cache_dir: str = "./data/processed"

    # ── Security ─────────────────────────────────────────────────────────────
    secret_key: str = "change-me-in-production"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": (),   # silence model_dir / model_type warnings
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
