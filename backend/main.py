"""
FastAPI application entry point.

Uncertainty-Guided Bayesian Active Learning for Materials Discovery
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.routes import datasets, evaluation, experiments
from backend.app.services.benchmark_service import recover_interrupted_benchmarks
from backend.config import get_settings
from backend.database import SessionLocal, create_all_tables

# ── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables and required directories on startup."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    create_all_tables()
    logger.info("Database tables ready (URL: %s)", settings.database_url[:30] + "...")
    with SessionLocal() as db:
        recovered = recover_interrupted_benchmarks(db)
    if recovered:
        logger.warning("Marked %d interrupted benchmark run(s) during startup", recovered)
    for d in [settings.upload_dir, settings.model_dir, settings.dataset_cache_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Application startup complete.")
    yield  # application runs here
    logger.info("Application shutdown.")


# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    description=(
        "Pool-based Bayesian active learning system for data-efficient materials discovery. "
        "Implements Gaussian Process surrogate models, Expected Improvement, and the "
        "PUDA hybrid acquisition strategy to iteratively select the most informative "
        "material candidates for virtual experimentation."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(evaluation.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health():
    return JSONResponse({"status": "ok", "version": settings.app_version})


@app.get("/", tags=["health"])
def root():
    return JSONResponse({
        "application": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    })
