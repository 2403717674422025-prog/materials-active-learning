export interface Dataset {
  id: string; name: string; source?: string
  num_samples?: number; num_features?: number
  target_column?: string; composition_column: string
  feature_columns: string[]; global_maximum?: number; global_minimum?: number
  status: 'pending' | 'ready' | 'error'
  upload_time: string; ready_time?: string; error_message?: string
}

export interface DatasetStatistics {
  n_samples: number; n_features: number
  target_min: number; target_max: number; target_mean: number
  target_std: number; target_median: number; target_q25: number; target_q75: number
  target_histogram: { counts: number[]; bin_edges: number[]; bin_centers: number[] }
  feature_names: string[]
  feature_means: number[]
  feature_stds: number[]
}

export type Strategy  = 'random' | 'uncertainty' | 'ei' | 'ucb' | 'puda'
export type ModelType = 'gp' | 'rf' | 'xgb' | 'ensemble'
export type ExpStatus = 'created' | 'initialized' | 'running' | 'completed' | 'error'

export interface ExperimentCreate {
  dataset_id: string; name?: string; strategy: Strategy; model_type: ModelType
  initial_samples: number; batch_size: number; total_budget: number; random_seed: number
  target_column: string; xi: number; kappa: number
  alpha: number; beta: number; gamma: number
  use_pca: boolean; pca_n_components: number
}

export interface TuningRun {
  id: string
  experiment_id: string
  model_type: 'rf' | 'gp'
  method: string
  objective: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  best_params?: Record<string, string | number | boolean | null> | null
  best_validation_rmse?: number | null
  best_validation_mae?: number | null
  best_validation_r2?: number | null
  trial_count: number
  completed_trials: number
  failed_trials: number
  pruned_trials: number
  random_seed: number
  timeout_seconds?: number | null
  training_size?: number | null
  validation_size?: number | null
  dataset_fingerprint?: string | null
  split_metadata?: Record<string, unknown> | null
  search_space_version: string
  trial_history: {
    number: number
    state: string
    value: number | null
    params: Record<string, string | number | boolean | null>
    error?: string
  }[]
  error_summary?: string | null
  applied_to_experiment: boolean
  created_at: string
  started_at?: string | null
  finished_at?: string | null
}

export interface Experiment {
  id: string; name?: string; dataset_id: string
  strategy: Strategy; model_type: ModelType
  initial_samples: number; batch_size: number; total_budget: number
  current_budget_used: number; current_iteration: number
  best_found_value?: number; best_found_id?: string; global_maximum?: number
  status: ExpStatus; created_at: string; updated_at: string
}

export interface ExperimentStatusResponse {
  experiment_id: string; status: ExpStatus; current_iteration: number
  budget_used: number; budget_remaining: number; budget_total: number; budget_pct: number
  best_found_value?: number; best_found_id?: string; global_maximum?: number
  latest_rmse?: number; latest_r2?: number; latest_regret?: number
}

export interface IterationResult {
  iteration: number; experiment_id: string; strategy: Strategy; model_type: ModelType
  budget_used: number; budget_remaining: number
  selected_candidate_ids: string[]; selected_true_values: number[]
  test_mae: number; test_rmse: number; test_r2: number
  best_found_value: number; best_found_id: string; regret: number; global_maximum: number
  training_pool_maximum: number
  mean_uncertainty_selected: number; diversity_score: number; iteration_time_seconds: number
}

export interface IterationHistoryItem {
  iteration: number; budget_used: number; budget_remaining: number
  test_mae?: number; test_rmse?: number; test_r2?: number
  best_found_value?: number; best_found_id?: string; regret?: number; diversity_score?: number
  n_candidates_selected: number; iteration_time_seconds?: number; timestamp: string
}

export interface CandidatePrediction {
  candidate_id: string; pool_index: number
  predicted_mean: number; predicted_std: number; lower_cb?: number | null; upper_cb?: number | null
  acquisition_score: number; is_selected: boolean; selection_rank?: number
  uncertainty_label: 'Low' | 'Medium' | 'High'; breakdown: Record<string, unknown>
}

export interface CandidateListResponse {
  experiment_id: string; iteration: number
  total_candidates: number; selected_count: number
  candidates: CandidatePrediction[]
  page: number; page_size: number; total_pages: number
}

export interface CandidateDetail {
  candidate_id: string; iteration: number
  predicted_mean: number; predicted_std: number; lower_cb?: number | null; upper_cb?: number | null
  strategy: string; acquisition_score: number
  predicted_performance?: number; uncertainty?: number
  exploration_contribution?: number; exploitation_contribution?: number; diversity_contribution?: number
  formula?: string; selection_reason: string
  is_selected: boolean; selection_rank?: number
  full_breakdown: Record<string, unknown>
}

export interface DiscoveredMaterial {
  rank: number; candidate_id: string; true_value: number
  iteration_discovered: number; strategy: string; budget_at_discovery: number
  predicted_mean: number; predicted_std: number; acquisition_score: number
  feature_preview: Record<string, number>
}

export interface StrategySeries {
  strategy: string; label: string; color: string; budget: number[]
  budget_unit?: 'cumulative_candidates' | 'training_candidates'; budget_label?: string
  values?: number[]
  std?: Array<number | null>
  n_runs?: number[]
  best_found?: number[]; best_found_std?: Array<number | null>
  rmse?: number[]; mae?: number[]; regret?: number[]; diversity?: number[]
}

export interface BenchmarkSummaryRow {
  rank: number | null
  strategy: string; label: string; color: string
  n_seeds: number; n_requested_seeds: number
  n_incomplete: number; n_failed: number
  naudc_mean: number | null; naudc_std: number | null
  top10_mean: number | null; top10_std: number | null
  final_mae_mean: number | null; final_mae_std: number | null
  final_rmse_mean: number | null; final_rmse_std: number | null
  final_regret_mean: number | null; final_regret_std: number | null
  actual_budget_mean: number | null; actual_budget_std: number | null
  runtime_seconds_mean: number | null; runtime_seconds_std: number | null
  global_maximum: number | null
}

export interface BenchmarkResult {
  experiment_id: string
  benchmark_id: string
  benchmark_status: 'completed' | 'partial' | 'failed'
  elapsed_seconds: number
  n_seeds: number
  seeds: number[]
  model_type: string
  strategies_run: string[]
  configuration: {
    protocol_version: string
    initial_samples: number; batch_size: number; total_budget: number; base_seed: number
    dataset_id: string; dataset_fingerprint: string
    model: Record<string, unknown>
    split: { random_state: number; pool_size: number; test_size_rows: number }
  }
  failed_or_incomplete_runs: Array<{
    strategy: string; seed: number; status: string; error?: string | null; actual_budget: number
  }>
  per_seed_results: Array<Record<string, unknown>>
  summary_table: BenchmarkSummaryRow[]
  discovery_curve: StrategySeries[]
  rmse_curve: StrategySeries[]
  mae_curve: StrategySeries[]
  regret_curve: StrategySeries[]
  diversity_curve: StrategySeries[]
  best_strategy: string | null
  global_maximum: number | null
  // ablation fields (same shape)
  ablation_variants?: string[]
  formulas?: Record<string, string>
  interpretation?: string
}

export interface CalibrationResult {
  experiment_id?: string; iteration?: number; n_aligned: number; n_observations?: number
  n_excluded?: number; exclusion_counts?: Record<string, number>; minimum_required?: number
  mae_on_observed?: number; rmse_on_observed?: number
  ece?: number; coverage_90pct?: number; coverage_95pct?: number
  reliability_diagram?: { expected_coverage: number[]; observed_coverage: number[] }
  interpretation?: string; caveat?: string
  error?: string
}

export interface ShapFeatureImportance {
  feature: string
  mean_abs_shap: number
}

export interface ShapBeeswarmPoint {
  candidate_id: string
  feature: string
  feature_value: number
  shap_value: number
}

export interface ShapGlobalExplanation {
  experiment_id: string
  model_type: string
  model_iteration: number
  baseline: number
  sample_size: number
  total_features: number
  feature_importance: ShapFeatureImportance[]
  beeswarm_points: ShapBeeswarmPoint[]
  interpretation: string
  feature_value_source: string
  reference_method: string
}

export interface ShapContribution {
  feature: string
  feature_value: number
  shap_value: number
}

export interface ShapCandidateExplanation {
  experiment_id: string
  candidate_id: string
  model_iteration: number
  predicted_value: number
  baseline: number
  contributions: ShapContribution[]
  largest_positive: ShapContribution[]
  largest_negative: ShapContribution[]
  interpretation: string
}

export interface ShapDependenceResult {
  experiment_id: string
  feature: string
  sample_size: number
  points: { candidate_id: string; feature_value: number; shap_value: number }[]
}

export interface EmbeddingPoint {
  candidate_id: string
  x: number; y: number; z?: number
  is_observed: boolean
  true_value?: number
  predicted_mean?: number
  predicted_std?: number
  acquisition_score?: number
  is_selected: boolean
}

export interface EmbeddingResult {
  experiment_id: string; iteration: number
  n_components: number; explained_variance: number[]; total_explained: number
  method: string; n_pool: number; n_shown: number; n_observed: number
  points: EmbeddingPoint[]
  axis_labels: string[]
}

export interface SummaryTableRow {
  rank: number; strategy: string; label: string; color: string
  naudc: number; final_rmse: number; final_r2: number; final_regret: number
  top10_discovery_rate: number; n_iterations: number
}

export interface ComparisonCharts {
  discovery_curve: StrategySeries[]; rmse_curve: StrategySeries[]
  regret_curve: StrategySeries[]; diversity_curve: StrategySeries[]
  summary_table: SummaryTableRow[]
  ranking: Record<string, unknown>; global_optimum: number; strategies: string[]
}
