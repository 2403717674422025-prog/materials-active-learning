# Uncertainty-Guided Bayesian Active Learning for Data-Efficient Materials Discovery

> A production-grade ML web application that intelligently selects which material candidates to evaluate next — discovering high-performing materials with substantially fewer experiments.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Research Question](#2-research-question)
3. [Motivation](#3-motivation)
4. [Dataset](#4-dataset)
5. [ML Methodology](#5-ml-methodology)
6. [Bayesian Modeling & Uncertainty](#6-bayesian-modeling--uncertainty)
7. [Active Learning Strategy](#7-active-learning-strategy)
8. [Acquisition Functions](#8-acquisition-functions)
9. [Experimental Simulation](#9-experimental-simulation)
10. [Baselines & Comparison](#10-baselines--comparison)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Architecture](#12-architecture)
13. [Repository Structure](#13-repository-structure)
14. [Installation — Local](#14-installation--local)
15. [Running with Docker](#15-running-with-docker)
16. [Deployment](#16-deployment)
17. [API Reference](#17-api-reference)
18. [Frontend Pages](#18-frontend-pages)
19. [Running Tests](#19-running-tests)
20. [Limitations](#20-limitations)
21. [Future Work](#21-future-work)
22. [Hyperparameter Optimization](#22-hyperparameter-optimization)
23. [License](#23-license)

---

## 1. Problem Statement

Discovering new high-performing materials (semiconductors, catalysts, alloys, thermoelectrics) requires evaluating candidates through computationally expensive simulations or physical experiments. A typical materials design space contains thousands to millions of candidates, but experimental budgets allow evaluating only a small fraction.

**The core challenge:** Given a large pool of unevaluated material candidates and a limited budget, which candidates should be evaluated to maximise discovery of high-performing materials?

This system answers that question using **pool-based Bayesian active learning** — a principled ML framework that uses predictive uncertainty to intelligently allocate the experimental budget.

---

## 2. Research Question

> "If I can only evaluate N material candidates in total, which N candidates should I choose to maximise the chance of discovering high-performing materials as early as possible?"

The system optimises simultaneously for:
- **Predictive accuracy** — how well the surrogate model estimates material properties
- **Uncertainty quantification** — knowing where the model is confident vs. uncertain
- **Exploration** — sampling uncertain regions to reduce epistemic uncertainty
- **Exploitation** — sampling regions with high predicted performance
- **Diversity** — avoiding redundant selections that provide no new information

---

## 3. Motivation

- Experimental evaluation of materials is expensive (days/weeks of compute or lab time)
- Random search wastes budget on uninformative candidates
- Pure exploitation (greedy) gets stuck in local optima
- Standard ML requires large labelled datasets — active learning is designed for the small-data regime
- No accessible, interactive web platform existed for this workflow

---

## 4. Dataset

| Property | Value |
|---|---|
| Name | `matbench_expt_gap` — Experimental Band Gap |
| Source | [matminer / Materials Project](https://hackingmaterials.lbl.gov/matminer/dataset_summary.html) |
| Paper | Zhuo et al., *J. Phys. Chem. Lett.* 9, 1668–1673 (2018) |
| License | **CC BY 4.0** |
| Samples | 4,604 inorganic semiconductors |
| Input | Chemical composition string (e.g. `GaAs`, `TiO2`) |
| Target | Experimental optical band gap [eV] — continuous |
| Missing values | None (pre-cleaned by Matbench) |

**Feature engineering** converts each composition string into ~144 numerical descriptors using matminer's `ElementProperty` (Magpie preset), `Stoichiometry`, `ValenceOrbital`, `IonProperty`, and `BandCenter` featurizers. Near-constant and highly correlated features are filtered from training rows only, typically leaving ~100–120 features.

---

## 5. ML Methodology

### Feature Pipeline

```
Composition string (e.g. "GaAs")
        ↓
pymatgen Composition object
        ↓
ElementProperty (Magpie)  → 132 features
Stoichiometry             →   5 features
ValenceOrbital            →   4 features
IonProperty               →   2 features
BandCenter                →   1 feature
        ↓
Fit near-constant filter on training rows (var < 1e-6)
Fit correlation filter on training rows (|r| > 0.95)
        ↓
StandardScaler — fit ONLY on labeled set at each iteration
        ↓
Optional PCA (auto-enabled for N_labeled > 800)
        ↓
~100–120 numerical features
```

**Critical invariant:** Feature filtering, imputation, StandardScaler, and optional PCA are fit at each active learning iteration using **only the current labeled set**. The already-fitted transforms are applied to the unlabeled pool and test set without refitting, preventing preprocessing leakage.

### Baseline Models

| Model | Uncertainty Method | Use |
|---|---|---|
| Random Forest (200 trees) | Std of individual tree predictions | Primary non-Bayesian baseline |
| XGBoost (bootstrap ensemble) | Std of 5 bootstrap models | Secondary comparison |

### Bayesian Surrogate

| Model | Uncertainty Method |
|---|---|
| Gaussian Process Regressor | True posterior standard deviation σ(x) |
| Deep Ensemble (5 MLPs) | Std of ensemble member predictions |

---

## 6. Bayesian Modeling & Uncertainty

### Gaussian Process Regression

The GP places a prior over functions:

```
f(x) ~ GP(m(x), k(x, x'))
```

**Kernel:**
```
k(x, x') = σ²_f · RBF(length_scale=l) + σ²_n · WhiteKernel()
```

After observing training data (X, y), the GP posterior is:

```
p(f* | x*, X, y) = N(μ(x*), σ²(x*))

μ(x*) = k(x*, X) [K + σ²_n I]⁻¹ y
σ²(x*) = k(x*, x*) - k(x*, X) [K + σ²_n I]⁻¹ k(X, x*)
```

**Key property:** σ(x) is the **true Bayesian posterior standard deviation** — not a heuristic score. It represents epistemic uncertainty: regions with few nearby training points have high σ.

Hyperparameters `σ²_f`, `l`, `σ²_n` are optimised by maximising the log-marginal-likelihood during each `fit()` call.

**Output per candidate:**
```json
{
  "predicted_property": 2.4812,
  "prediction_std": 0.3241,
  "lower_confidence_bound": 1.8460,
  "upper_confidence_bound": 3.1164
}
```

---

## 7. Active Learning Strategy

### Pool-Based Active Learning Loop

```
Fixed held-out test set (20%)
        ↑ (never touched during AL loop)

Active Learning Pool (80%)
        │
        ├── Labeled set  (starts: 100 samples)
        │
        └── Unlabeled pool (starts: ~3,580 samples)
                │
                ▼
     ┌─── Train surrogate model ────────────────┐
     │    (GP or RF on labeled set)             │
     │                                          │
     ▼                                          │
Predict entire unlabeled pool → (μ, σ)         │
     │                                          │
     ▼                                          │
Calculate acquisition scores                    │
     │                                          │
     ▼                                          │
Select batch (batch_size=20)                   │
with diversity-aware greedy selection           │
     │                                          │
     ▼                                          │
Simulated experimental evaluation              │
(reveal true dataset value)                    │
     │                                          │
     ▼                                          │
Add to labeled set ──────────────────────────┘
     │
     ▼
Evaluate on test set (MAE, RMSE, R²)
Track best-found, regret, diversity
     │
     ▼
Repeat until budget exhausted
```

### Engine Abstraction

```python
engine = ActiveLearningEngine(config)
state  = engine.initialize(pool_X, pool_y, pool_ids, test_X, test_y, test_ids)

for iteration in range(n_iterations):
    result, state = engine.run_iteration(state)
    # result contains: predictions, selections, metrics, breakdowns
```

---

## 8. Acquisition Functions

### Strategy 1 — Random Sampling

```
score(x) = Uniform(0, 1)
```
Selects candidates uniformly at random. No model information used. Lower bound on all other strategies.

### Strategy 2 — Uncertainty Sampling

```
score(x) = σ(x)
```
Selects candidates with highest predictive uncertainty. Pure exploration; ignores predicted performance.

### Strategy 3 — Expected Improvement (Bayesian Optimisation)

```
EI(x) = (μ(x) - y* - ξ) · Φ(Z) + σ(x) · φ(Z)

where:
  Z     = (μ(x) - y* - ξ) / σ(x)
  y*    = max observed value so far
  ξ     = exploration jitter (default: 0.01)
  Φ, φ  = standard normal CDF/PDF
```

EI naturally balances exploration (σ·φ(Z) term) and exploitation (improvement over y*). When σ→0, EI→max(μ-y*, 0). When σ is large, the integral over possible improvements is substantial even when μ < y*.

### Strategy 4 — PUDA (Proposed Hybrid)

```
PUDA(x) = α · norm(μ(x))
         + β · norm(σ(x))
         + γ · norm(div(x, S_selected))

div(x, S) = min_{s ∈ S} cosine_distance(x, s)

Default weights:  α=0.40, β=0.35, γ=0.25
```

**Greedy batch selection:** Candidates are selected one by one. After each pick, the selected candidate joins `S_selected` and diversity scores are recomputed, implementing a greedy k-center approximation.

**Why PUDA addresses EI's weakness:** EI and UCB select the top-k candidates independently, which can yield a batch of near-identical candidates. PUDA's diversity term explicitly penalises redundant selections. The diversity component's contribution is validated by the ablation study.

**Ablation variants:**

| Variant | α | β | γ |
|---|---|---|---|
| Prediction only | 1.0 | 0 | 0 |
| Pred + Uncertainty | 0.5 | 0.5 | 0 |
| Pred + Diversity | 0.5 | 0 | 0.5 |
| Full PUDA | 0.4 | 0.35 | 0.25 |

---

## 9. Experimental Simulation

Because real laboratory experiments cannot be performed in software, the system uses a **transparent retrospective benchmark**:

```
Model selects candidate "Fe2O3" (pool index 847)
        ↓
Simulator retrieves: df.loc[847, "gap expt"] = 2.36 eV
        ↓
Returns 2.36 eV as the "experimental observation"
        ↓
Added to labeled training set
        ↓
Model retrains
```

**The model never sees the true value before it selects the candidate.** The value is withheld until selection, exactly as in a real experimental campaign.

This is clearly labelled throughout the UI as **"Simulated Experimental Evaluation"** with an explanation that this is a retrospective benchmark — not a real laboratory result.

---

## 10. Baselines & Comparison

The system runs controlled experiments comparing:

| Strategy | Description |
|---|---|
| Random | Uniform random sampling — no ML |
| Uncertainty | Highest GP posterior σ(x) |
| EI | Expected Improvement (ξ=0.01) |
| PUDA | Proposed hybrid (α=0.4, β=0.35, γ=0.25) |

All strategies use identical datasets, train/test splits, model types, and initial labeled candidates within each seed. The documented default seeds are 42, 123, and 456. Benchmark snapshots preserve dataset fingerprints, split IDs, initial and acquired candidate IDs, per-seed metrics, runtime, and incomplete/failed statuses without modifying parent experiment history.

Held-out test targets are used only for predictive evaluation after each iteration and at the final budget, not for acquisition, tuning, or strategy ranking. The retrospective discovery reference and NAUDC normalization use the training pool. Discovery curves use cumulative candidate counts (including partial batches); predictive-error curves use the training-set size used to fit the model. Across-seed standard deviations use the sample estimate; one-seed variability is not estimable.

### Designed Experiments

| Experiment | Compares |
|---|---|
| A | Random vs. all active strategies |
| B | Uncertainty sampling vs. EI |
| C | EI vs. PUDA |
| D | Budget size: 50 / 100 / 200 / 500 experiments |
| E | Seed size: 20 / 50 / 100 / 200 initial samples |

---

## 11. Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | (1/N) Σ\|y-ŷ\| | Prediction accuracy |
| RMSE | √((1/N) Σ(y-ŷ)²) | Prediction accuracy |
| R² | 1 - SS_res/SS_tot | Explained variance |
| Best-Found | max(y_acquired[0:n]) | Discovery efficiency |
| Regret | y_global_max - best_found | Distance from optimum |
| NAUDC | ∫ best_found(t)/y_pool_max dt / T | Single-number budget efficiency |
| Top-10 Rate | \|found ∩ true_top10\| / 10 | Coverage of best materials |
| ECE | E[\|observed_coverage - expected_coverage\|] | Uncertainty calibration |
| Diversity | Mean pairwise cosine distance in batch | Batch coverage |

---

## 12. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React + Vite)                 │
│  Overview · Dataset · Setup · AL Dashboard · Candidates         │
│  Explainer · History · Compare · Discovery                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (REST JSON)
                             │ /api/*
┌────────────────────────────▼────────────────────────────────────┐
│                       FastAPI Backend                           │
│                                                                 │
│  Routes: /datasets · /experiments · /evaluation                 │
│  Services: dataset_service · experiment_service · eval_service  │
│  Schemas: Pydantic v2 request/response validation               │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    ML Pipeline                            │  │
│  │                                                           │  │
│  │  feature_engineering.py  →  preprocessing.py             │  │
│  │         ↓                        ↓                        │  │
│  │  GaussianProcessSurrogate   RandomForestSurrogate         │  │
│  │         ↓                        ↓                        │  │
│  │  ExpectedImprovement    UncertaintyAcquisition            │  │
│  │  PUDAAcquisition        RandomAcquisition                 │  │
│  │         ↓                                                 │  │
│  │  ActiveLearningEngine.run_iteration()                     │  │
│  │         ↓                                                 │  │
│  │  metrics.py · comparison.py · discovery.py                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ORM: SQLAlchemy 2.0 → PostgreSQL / SQLite                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Repository Structure

```
materials-active-learning/
│
├── frontend/
│   ├── src/
│   │   ├── pages/              9 page components
│   │   ├── components/         Shared UI (BudgetMeter, AcquisitionBreakdown, ...)
│   │   ├── services/api.ts     Axios API client
│   │   ├── types/index.ts      TypeScript interfaces
│   │   └── index.css           Tailwind + design tokens
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── vite.config.ts
│
├── backend/
│   ├── main.py                 FastAPI app entry point
│   ├── config.py               Pydantic-settings configuration
│   ├── database.py             SQLAlchemy engine + session
│   ├── requirements.txt
│   ├── Dockerfile
│   │
│   ├── app/
│   │   ├── routes/             datasets · experiments · evaluation
│   │   ├── services/           dataset · experiment · evaluation services
│   │   ├── schemas/            Pydantic request/response models
│   │   └── models/             SQLAlchemy ORM models
│   │
│   ├── ml/
│   │   ├── data/               feature_engineering · preprocessing
│   │   ├── models/             baseline (RF/XGB) · bayesian (GP/Ensemble)
│   │   ├── acquisition/        random · uncertainty · EI/UCB · PUDA
│   │   ├── active_learning/    engine.py (core AL loop)
│   │   └── evaluation/         metrics · comparison · discovery
│   │
│   └── tests/
│       ├── test_preprocessing.py
│       ├── test_models.py
│       ├── test_acquisition.py
│       ├── test_engine.py
│       ├── test_evaluation.py
│       └── test_api.py
│
├── data/
│   ├── raw/                    Downloaded at runtime (.gitignored)
│   ├── processed/              Feature cache (.gitignored)
│   └── README.md               Dataset documentation
│
├── experiments/                Saved experiment configs/results
├── docs/                       Architecture & methodology docs
│
├── docker-compose.yml          Production (Postgres + Backend + Frontend)
├── docker-compose.dev.yml      Development (SQLite, hot-reload)
├── .env.example                Environment variable template
├── .gitignore
└── README.md
```

---

## 14. Installation — Local

### Prerequisites

- Python 3.12+
- Node.js 20+
- Git

### Backend Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/materials-active-learning.git
cd materials-active-learning

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
cd backend
pip install -r requirements.txt

# 4. Configure environment
cp ../.env.example ../.env
# Edit .env: set DATABASE_URL (default: sqlite:///./materials_al.db)

# 5. Start backend
cd ..
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Backend is available at http://localhost:8000
API docs at http://localhost:8000/docs

### Frontend Setup

```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Configure API URL (development proxy is set in vite.config.ts)
# No .env needed for local dev — vite proxies /api → localhost:8000

# Start dev server
npm run dev
```

Frontend available at http://localhost:5173

### Load Dataset (First Run)

After both services are running, load the dataset via the UI:
1. Open http://localhost:5173
2. Navigate to **Dataset Explorer**
3. Click **Load expt_gap**

Or via API:
```bash
curl -X POST http://localhost:8000/datasets/load-builtin \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "matbench_expt_gap", "target_column": "gap expt"}'
```

The first load takes 3–10 minutes (matminer featurizes 4,604 compositions). Subsequent loads use cache and return instantly.

---

## 15. Running with Docker

### Full Stack (recommended)

```bash
# 1. Copy environment file
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD and SECRET_KEY

# 2. Start all services
docker compose up --build

# Services:
#   Frontend:  http://localhost:80
#   Backend:   http://localhost:8000
#   API Docs:  http://localhost:8000/docs
#   Postgres:  localhost:5432
```

### Development Mode (SQLite, hot-reload)

```bash
docker compose -f docker-compose.dev.yml up --build
# Backend: http://localhost:8000 (auto-reloads on code changes)
```

### Teardown

```bash
docker compose down         # stop containers, keep data volumes
docker compose down -v      # stop containers AND remove all data
```

---

## 16. Deployment

### Frontend → Vercel

```bash
cd frontend
npm run build            # produces frontend/dist/

# Deploy to Vercel
npx vercel --prod
# Set env var: VITE_API_URL=https://your-backend.railway.app
```

### Backend → Railway / Render

1. Connect GitHub repository
2. Set root directory to `backend/`
3. Set environment variables:
   ```
   DATABASE_URL=postgresql://...
   SECRET_KEY=<long-random-string>
   ALLOWED_ORIGINS=https://your-frontend.vercel.app
   ```
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### Computational Requirements

The core system runs on CPU. No GPU required.

| Component | Requirement |
|---|---|
| GP on N=100 labeled | < 1s CPU |
| GP on N=500 labeled | ~5s CPU |
| RF on N=500 labeled | < 1s CPU |
| Memory (backend) | ~512 MB |
| Memory (frontend) | ~64 MB (nginx) |
| Storage (dataset cache) | ~50 MB |

For N_labeled > 800, the system automatically switches to RF uncertainty or enables PCA reduction (configured via the Setup page).

---

## 17. API Reference

Full interactive docs at `/docs` (Swagger UI) and `/redoc`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/datasets/` | List all datasets |
| `POST` | `/datasets/load-builtin` | Load matbench dataset |
| `POST` | `/datasets/upload` | Upload CSV dataset |
| `GET` | `/datasets/{id}` | Dataset info |
| `GET` | `/datasets/{id}/statistics` | Feature & target stats |
| `GET` | `/datasets/{id}/preview` | First N rows |
| `POST` | `/experiments/` | Create experiment |
| `POST` | `/experiments/{id}/initialize` | Seed labeled set |
| `POST` | `/experiments/{id}/run-iteration` | Run one AL iteration |
| `POST` | `/experiments/{id}/tuning-runs` | Start RF/GP Optuna tuning before the first AL iteration |
| `GET` | `/experiments/{id}/tuning-runs` | List persisted tuning runs |
| `GET` | `/experiments/{id}/tuning-runs/{run_id}` | Read tuning progress, metrics, and trials |
| `POST` | `/experiments/{id}/tuning-runs/{run_id}/apply` | Apply best validation-selected parameters |
| `GET` | `/experiments/{id}/status` | Budget, best-found, RMSE |
| `GET` | `/experiments/{id}/history` | Per-iteration metrics |
| `GET` | `/experiments/{id}/candidates` | Pool predictions (paginated) |
| `GET` | `/experiments/{id}/candidates/{cid}` | Selection breakdown |
| `GET` | `/experiments/{id}/discovery` | Best discovered materials |
| `POST` | `/experiments/{id}/export` | Export results as JSON |
| `GET` | `/evaluation/{id}` | Experiment evaluation summary |
| `GET` | `/evaluation/compare/strategies` | Multi-strategy comparison |

---

## 18. Frontend Pages

| Page | Route | Description |
|---|---|---|
| Overview | `/` | Project description, quick-start, system summary |
| Dataset Explorer | `/dataset` | Load/upload datasets, target distribution, data preview |
| Experiment Setup | `/setup` | Configure budget, strategy, model, PUDA weights, optional model tuning |
| AL Dashboard | `/experiment/:id` | Live budget meter, RMSE curve, best-found tracking, tuning results |
| Candidate Explorer | `/experiment/:id/candidates` | Sortable/filterable table with predictions & scores |
| Selection Explainer | `/experiment/:id/candidates/:cid` | Per-candidate PUDA breakdown, why selected |
| Experiment History | `/experiment/:id/history` | Full per-iteration metrics table |
| Strategy Comparison | `/experiment/:id/compare` | Discovery/RMSE/regret/diversity curves, summary table |
| Material Discovery | `/experiment/:id/discovery` | Ranked list of discovered materials |

---

## 19. Running Tests

```bash
cd backend

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_acquisition.py -v
pytest tests/test_engine.py -v
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"
```

**Test coverage:**
- `test_preprocessing.py` — 8 tests: split, scaler, PCA, no-leakage, serialization
- `test_models.py` — 11 tests: RF and GP shape, uncertainty, calibration, save/load
- `test_acquisition.py` — 18 tests: all 4 strategies, budget, diversity, no-duplicates
- `test_engine.py` — 7 tests: all strategies, budget respected, no duplicates across iterations
- `test_evaluation.py` — 14 tests: metrics, NAUDC, regret, calibration, comparison ranking
- `test_api.py` — 12 tests: full API integration including iteration cycle
- `test_tuning.py` — Optuna search reproducibility, real RF/GP tuning, target-leakage guard, failures, and configuration application

---

## 20. Limitations

1. **GP scalability**: Gaussian Process has O(N³) training complexity. For labeled sets > ~800 samples, the system switches to RF. This is mitigated by PCA but is an inherent GP limitation.

2. **Sequential (not parallel) iteration**: Each iteration trains a new model from scratch. Incremental GP updates are not implemented, making each iteration O(N³) in the labeled set size.

3. **No true experimental validation**: All "experimental results" are retrospective lookups from the held-out dataset. The system benchmarks active learning efficiency, not real laboratory outcomes.

4. **Single-objective only (Phase 1)**: The current system optimises one property. Multi-objective Pareto-front discovery is designed but not yet implemented.

5. **Composition-only features**: The featurization uses elemental/compositional descriptors only. Crystal structure information (if available) would improve predictions but is not yet used.

6. **Uncertainty calibration varies by regime**: GP calibration degrades when the labeled set is very small (< 30 samples) or when training and test distributions differ significantly.

---

## 21. Future Work

- **Crystal structure features**: Integrate graph neural network representations for structure-aware predictions
- **Multi-objective extension**: Pareto-front active learning for simultaneous optimisation of multiple properties
- **Batch EI (qEI)**: Replace greedy batch EI with the true batch expected improvement via Monte Carlo integration
- **Incremental GP updates**: Woodbury identity-based rank-1 updates to avoid O(N³) full retraining
- **Transfer learning**: Pre-trained embeddings from large materials databases as feature initialisation
- **Real experiment integration**: API hook to submit candidates to real computational pipelines (VASP, Quantum ESPRESSO)
- **Uncertainty decomposition**: Separate epistemic (model) and aleatoric (data noise) uncertainty
- **AutoML surrogate selection**: Automatically choose the best surrogate model based on the current labeled set size

---

## 22. Hyperparameter Optimization

Bayesian active learning and Bayesian hyperparameter optimization are separate
tasks. The active-learning strategy chooses **which material to evaluate next**;
the optional Optuna tuning step chooses **model parameters** for Random Forest
or Gaussian Process Regression.

Enable **Optimize model hyperparameters with Optuna** on the Experiment Setup
page. Tuning runs before the first active-learning iteration and uses only the
experiment's initial labeled samples. Those samples are split reproducibly into
training and validation subsets (80/20, using the configured random seed).
Every trial fits its scaler/PCA on the training subset only and minimizes
validation RMSE. The held-out test set is not used to choose parameters.

The default tuning budget is 20 trials with a 15-minute timeout; the UI allows
1–100 trials. Optuna's seeded TPE sampler searches a bounded RF space
(`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and
`max_features`) or a practical GP space (`length_scale`, `noise_level`,
`n_restarts_optimizer`, and `normalize_y`). Run metadata, best validation
metrics, trial status/history, split metadata, and the feature-only dataset
fingerprint are persisted in the additive `tuning_runs` table. Existing
experiments keep their manual defaults; a completed run must be explicitly
applied before the first active-learning iteration.
The table is created by the existing SQLAlchemy `create_all` startup path; no
columns are added to or rewritten in the legacy experiment table.

The API uses `POST /experiments/{id}/tuning-runs` to start a background run,
`GET /experiments/{id}/tuning-runs/{run_id}` to poll progress, and
`POST /experiments/{id}/tuning-runs/{run_id}/apply` to activate the selected
configuration. The dashboard shows validation RMSE and trial history
separately from held-out test metrics.

Tuning can be computationally expensive, particularly for Gaussian Processes,
whose fit scales cubically in the number of training samples. These bounded
search spaces and trial/time limits are intended for a student research
workflow, not a guarantee of globally optimal parameters. Lower validation
RMSE does **not** guarantee improved held-out test performance. For strategy
comparisons, use the same model, trial budget, and tuning protocol for every
acquisition strategy; do not tune against test results.

No historical benchmark outcomes are generated by tuning itself. Run and save
the desired experiments before reporting performance comparisons.

## 23. License

This project is released under the **MIT License**. See `LICENSE` for full text.

The `expt_gap` dataset used in this project is licensed under **CC BY 4.0**.
Attribution: Zhuo et al., *J. Phys. Chem. Lett.* 2018, 9, 1668–1673.
https://creativecommons.org/licenses/by/4.0/

---

*Built as a genuine Advanced Machine Learning project demonstrating pool-based Bayesian active learning for materials discovery. All experimental results are retrospective benchmarks — no real laboratory experiments were conducted.*
