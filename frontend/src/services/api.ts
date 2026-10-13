import axios from 'axios'
import type {
  Dataset, DatasetStatistics, Experiment, ExperimentCreate,
  ExperimentStatusResponse, IterationResult, IterationHistoryItem,
  CandidateListResponse, CandidateDetail, DiscoveredMaterial, ComparisonCharts,
  BenchmarkResult, CalibrationResult, EmbeddingResult,
  TuningRun,
  ShapGlobalExplanation, ShapCandidateExplanation, ShapDependenceResult,
} from '../types'

const BASE = '/api'
export const http = axios.create({ baseURL: BASE, timeout: 300_000 })  // 5 min for benchmarks

export const api = {
  datasets: {
    list:       ()                       => http.get<Dataset[]>('/datasets/').then(r => r.data),
    get:        (id: string)             => http.get<Dataset>(`/datasets/${id}`).then(r => r.data),
    loadBuiltin:(name: string, col: string) =>
      http.post('/datasets/load-builtin', { dataset_name: name, target_column: col }).then(r => r.data),
    upload:     (form: FormData)         =>
      http.post('/datasets/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data),
    preview:    (id: string, n = 100)    => http.get(`/datasets/${id}/preview?n_rows=${n}`).then(r => r.data),
    statistics: (id: string)             => http.get<DatasetStatistics>(`/datasets/${id}/statistics`).then(r => r.data),
  },

  experiments: {
    list:           (skip = 0, limit = 50) =>
      http.get<Experiment[]>(`/experiments/?skip=${skip}&limit=${limit}`).then(r => r.data),
    create:         (p: ExperimentCreate)  => http.post<Experiment>('/experiments/', p).then(r => r.data),
    initialize:     (id: string)           => http.post(`/experiments/${id}/initialize`).then(r => r.data),
    runIteration:   (id: string)           => http.post<IterationResult>(`/experiments/${id}/run-iteration`).then(r => r.data),
    startTuning:    (id: string, input: { model_type: 'rf' | 'gp'; n_trials: number; random_seed: number; timeout_seconds: number }) =>
      http.post<TuningRun>(`/experiments/${id}/tuning-runs`, input).then(r => r.data),
    tuningRuns:     (id: string) =>
      http.get<TuningRun[]>(`/experiments/${id}/tuning-runs`).then(r => r.data),
    tuningStatus:   (id: string, runId: string) =>
      http.get<TuningRun>(`/experiments/${id}/tuning-runs/${runId}`).then(r => r.data),
    applyTuning:    (id: string, runId: string) =>
      http.post<TuningRun>(`/experiments/${id}/tuning-runs/${runId}/apply`).then(r => r.data),
    status:         (id: string)           => http.get<ExperimentStatusResponse>(`/experiments/${id}/status`).then(r => r.data),
    history:        (id: string)           => http.get<IterationHistoryItem[]>(`/experiments/${id}/history`).then(r => r.data),
    candidates:     (id: string, iter?: number, page = 1, size = 50) =>
      http.get<CandidateListResponse>(
        `/experiments/${id}/candidates?page=${page}&page_size=${size}` + (iter != null ? `&iteration=${iter}` : ''),
      ).then(r => r.data),
    candidateDetail:(expId: string, cid: string) =>
      http.get<CandidateDetail>(`/experiments/${expId}/candidates/${encodeURIComponent(cid)}`).then(r => r.data),
    discovery:      (id: string, n = 50)   => http.get<DiscoveredMaterial[]>(`/experiments/${id}/discovery?top_n=${n}`).then(r => r.data),
    export:         (id: string)           => http.post(`/experiments/${id}/export`).then(r => r.data),
    // New endpoints
    embedding:      (id: string, nComponents = 2) =>
      http.get<EmbeddingResult>(`/experiments/${id}/embedding?n_components=${nComponents}`).then(r => r.data),
    benchmark:      (id: string, opts?: { seeds?: string; model_type?: string; strategies?: string[] }) => {
      const params = new URLSearchParams()
      if (opts?.seeds) params.append('seeds', opts.seeds)
      if (opts?.model_type) params.append('model_type', opts.model_type)
      if (opts?.strategies) opts.strategies.forEach(s => params.append('strategies', s))
      return http.post<BenchmarkResult>(`/experiments/${id}/benchmark?${params}`).then(r => r.data)
    },
    getBenchmark:   (id: string, benchmarkId: string) =>
      http.get<Record<string, unknown>>(
        `/experiments/${id}/benchmarks/${benchmarkId}`,
      ).then(r => r.data),
    ablation:       (id: string, opts?: { seeds?: string; model_type?: string }) => {
      const params = new URLSearchParams()
      if (opts?.seeds) params.append('seeds', opts.seeds)
      if (opts?.model_type) params.append('model_type', opts.model_type)
      return http.post<BenchmarkResult>(`/experiments/${id}/ablation?${params}`).then(r => r.data)
    },
    calibration:    (id: string)           => http.get<CalibrationResult>(`/experiments/${id}/calibration`).then(r => r.data),
    explainability: {
      candidates: (id: string, q: string) =>
        http.get<{ candidates: string[]; total_matches: number }>(
          `/experiments/${id}/explainability/candidates`, { params: { q, limit: 50 } },
        ).then(r => r.data),
      global: (id: string) =>
        http.get<ShapGlobalExplanation>(`/experiments/${id}/explainability/global`).then(r => r.data),
      candidate: (id: string, candidateId: string) =>
        http.get<ShapCandidateExplanation>(
          `/experiments/${id}/explainability/candidates/${encodeURIComponent(candidateId)}`,
        ).then(r => r.data),
      dependence: (id: string, feature: string) =>
        http.get<ShapDependenceResult>(`/experiments/${id}/explainability/dependence`, {
          params: { feature },
        }).then(r => r.data),
    },
    report:         (id: string, fmt: 'json' | 'csv' = 'json') =>
      http.get(`/experiments/${id}/report?fmt=${fmt}`, { responseType: 'blob' }).then(r => r.data),
  },

  evaluation: {
    summary: (id: string)    => http.get(`/evaluation/${id}`).then(r => r.data),
    compare: (ids: string[]) =>
      http.get<ComparisonCharts>(
        `/evaluation/compare/strategies?${ids.map(i => `experiment_ids=${i}`).join('&')}`
      ).then(r => r.data),
  },
}
