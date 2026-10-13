# System Architecture Overview

## Component Map

```
Browser
  └── React SPA (Vite, Tailwind, Recharts)
        └── API calls via axios (/api/*)
              └── Nginx reverse proxy (port 80)
                    └── FastAPI backend (port 8000)
                          ├── Routes: /datasets, /experiments, /evaluation
                          ├── Services: dataset, experiment, evaluation
                          ├── ML Pipeline
                          │     ├── Feature Engineering (matminer)
                          │     ├── Preprocessing (train-fitted filters, imputation, scaler, PCA)
                          │     ├── Surrogates (GP, RF, Ensemble)
                          │     ├── Acquisition (Random, Uncertainty, EI, PUDA)
                          │     ├── Engine (AL loop)
                          │     └── Evaluation (metrics, comparison)
                          └── Database (PostgreSQL / SQLite)
                                ├── datasets
                                ├── experiments
                                ├── experiment_iterations
                                ├── candidate_predictions
                                └── experimental_results
```

## Data Flow — One AL Iteration

```
POST /experiments/{id}/run-iteration
    │
    ├── Load labeled set from DB + state cache
    ├── fit_preprocessor(labeled_X)         ← filters, imputation, scaler/PCA on labeled only
    ├── transform(labeled_X, unlabeled_X, test_X) ← reuse fitted rules; no refit
    ├── model.fit(X_labeled_scaled, y_labeled)
    ├── mean, std = model.predict(X_unlabeled_scaled)
    ├── scores = acquisition.score(mean, std, X_unlabeled)
    ├── selected_idx = greedy_batch_select(scores, batch_size)
    ├── true_values = simulator.acquire(selected_idx, pool_y)
    ├── state.labeled_indices += selected_idx
    ├── metrics = model.evaluate(X_test_scaled, y_test)
    ├── persist(ExperimentIteration, CandidatePredictions, ExperimentalResults)
    └── return IterationResult
```

## Database Schema (simplified)

```
datasets ─────────────────────── experiments
  id (PK)                          id (PK)
  name                             dataset_id (FK)
  num_samples                      strategy
  num_features                     model_type
  target_column                    initial_samples / batch_size / total_budget
  cache_path                       current_iteration / current_budget_used
  status                           ─────────────────────────────────
                                        │
                                   experiment_iterations
                                     experiment_id (FK)
                                     iteration_num
                                     test_mae / rmse / r2
                                     best_found_value / regret
                                     ──────────────────────
                                   candidate_predictions
                                     experiment_id (FK)
                                     iteration_num
                                     predicted_mean / std
                                     acquisition_score
                                     is_selected / breakdown
                                     ──────────────────────
                                   experimental_results
                                     experiment_id (FK)
                                     iteration_num
                                     candidate_id
                                     true_value  ← simulator output
```
