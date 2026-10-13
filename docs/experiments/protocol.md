# Experiment Protocol

## Setup (identical for all experiments)

- Dataset: matbench_expt_gap (4,604 samples)
- Test set: 20% (fixed, held out for all iterations)
- Active learning pool: 80% (3,683 samples)
- Surrogate model: Gaussian Process (RBF + White kernel)
- Metrics: MAE, RMSE, R², best-found, regret, NAUDC, diversity
- Repetitions: 3 random seeds (42, 123, 456) — report mean ± sample std
- Keep the train/test split fixed across all seeds using the experiment's split
  seed. Within each seed, use the same initial labeled candidate indices for all
  four acquisition strategies. For one completed seed, variability is not
  estimable; do not report a zero standard deviation.
- Use the same model type and fixed model parameters for every strategy. The
  standard benchmark uses the defaults (RF: 100 trees; GP: 3 optimizer restarts)
  and does not tune against the test set.
- Persist the dataset fingerprint, split IDs and seed, initial candidate IDs,
  acquisition order, configuration, per-seed curves/metrics, status, and
  runtime without modifying historical experiment records.
- Optional model tuning: if enabled, use the same Optuna model, trial budget,
  search-space version, and validation protocol for every acquisition strategy.
  Tune from each experiment's initial labeled training-pool samples only; never
  use held-out test results to select hyperparameters.

Hyperparameter optimization selects model parameters; active learning selects
material candidates. Record validation RMSE separately from test RMSE and do
not claim tuning improves test performance without a completed evaluation.

The test set is used only for held-out predictive evaluation (iteration curves
and the final budget model). It is not passed into acquisition scoring, tuning,
or strategy ranking. Discovery
reference values and NAUDC normalization use the training pool in this
retrospective benchmark. Discovery curves label every cumulative candidate
count; predictive-error curves use the training-set size used to fit the model,
before the next batch is acquired. Partial batches end at their actual count.

## Experiment A — Active Learning vs Random Baseline
- Initial samples: 100
- Batch size: 20
- Total budget: 300
- Strategies: Random vs. Uncertainty vs. EI vs. PUDA

## Experiment B — Uncertainty vs Bayesian Acquisition
- Initial samples: 100
- Batch size: 20
- Total budget: 300
- Strategies: Uncertainty vs. EI

## Experiment C — EI vs PUDA
- Initial samples: 100
- Batch size: 20
- Total budget: 300
- Strategies: EI vs. PUDA (full) vs. PUDA ablation variants

## Experiment D — Budget Sensitivity
- Strategy: PUDA (best strategy from Experiment A)
- Initial samples: 50
- Batch size: 20
- Total budgets: 50, 100, 200, 500

## Experiment E — Seed Size Sensitivity
- Strategy: PUDA
- Batch size: 20
- Total budget: 300
- Initial samples: 20, 50, 100, 200

## Ablation Study (Experiment C extension)
| Variant | α | β | γ |
|---|---|---|---|
| Prediction only | 1.0 | 0 | 0 |
| Pred + Unc | 0.5 | 0.5 | 0 |
| Pred + Div | 0.5 | 0 | 0.5 |
| Full PUDA | 0.4 | 0.35 | 0.25 |

Primary metric for ranking: NAUDC (Normalized Area Under Discovery Curve)
Secondary metrics: final RMSE, final regret, top-10 discovery rate
