import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Play, Plus, X, Trophy, RefreshCw, Info, AlertTriangle } from 'lucide-react'
import { api } from '../services/api'
import type { BenchmarkResult } from '../types'
import DiscoveryCurveChart from '../components/DiscoveryCurveChart'
import EmptyState from '../components/EmptyState'
import clsx from 'clsx'

const SM: Record<string, { label: string; color: string }> = {
  random:      { label: 'Random Sampling',     color: '#3A4F64' },
  uncertainty: { label: 'Uncertainty Sampling', color: '#59D5CC' },
  ei:          { label: 'Expected Improvement', color: '#A99BFF' },
  puda:        { label: 'PUDA (Proposed)',       color: '#41D98C' },
  pred_only:   { label: 'Prediction Only',       color: '#F06565' },
  pred_unc:    { label: 'Pred + Uncertainty',    color: '#EDAB60' },
  pred_div:    { label: 'Pred + Diversity',      color: '#59D5CC' },
  full:        { label: 'Full PUDA',             color: '#41D98C' },
}

function formatMeanStd(
  mean: number | null,
  std: number | null,
  digits = 4,
): string {
  if (mean == null) return '—'
  return `${mean.toFixed(digits)}${std == null ? '' : ` ±${std.toFixed(digits)}`}`
}

export default function StrategyComparisonPage() {
  const { id } = useParams<{ id: string }>()
  const [tab, setTab] = useState<'benchmark' | 'ablation' | 'calibration'>('benchmark')

  // Benchmark state
  const [seeds, setSeeds] = useState('42,123,456')
  const [modelType, setModelType] = useState<'rf' | 'gp'>('rf')
  const [benchResult, setBenchResult] = useState<BenchmarkResult | null>(null)
  const [ablResult, setAblResult] = useState<BenchmarkResult | null>(null)

  const bench = useMutation({
    mutationFn: () => api.experiments.benchmark(id!, { seeds, model_type: modelType }),
    onSuccess: d => setBenchResult(d as BenchmarkResult),
  })
  const abl = useMutation({
    mutationFn: () => api.experiments.ablation(id!, { seeds, model_type: modelType }),
    onSuccess: d => setAblResult(d as BenchmarkResult),
  })

  const { data: cal, isLoading: calLoading } = useQuery({
    queryKey: ['calibration', id],
    queryFn: () => api.experiments.calibration(id!),
    enabled: tab === 'calibration' && !!id,
  })

  function buildSeries(result: BenchmarkResult | null, key: 'discovery_curve' | 'rmse_curve' | 'regret_curve' | 'diversity_curve') {
    if (!result) return []
    const raw = result[key] as any[]
    return raw.map(s => ({
      ...s,
      values: s[key === 'discovery_curve' ? 'best_found' : key === 'rmse_curve' ? 'rmse' : key === 'regret_curve' ? 'regret' : 'diversity'] ?? [],
      std: s[key === 'discovery_curve' ? 'best_found_std' : key === 'rmse_curve' ? 'rmse_std' : key === 'regret_curve' ? 'regret_std' : 'diversity_std'],
    }))
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-5 py-3.5 border-b border-line2 bg-ink1 flex items-center justify-between">
        <div>
          <div className="t-title text-[13px]">Strategy Benchmark</div>
          <div className="text-t3 text-[11px] mt-0.5">
            Compare acquisition strategies · Ablation study · Uncertainty calibration
          </div>
        </div>
        {/* Tab switcher */}
        <div className="flex items-center gap-0.5 bg-ink border border-line2 rounded p-0.5">
          {[
            { k: 'benchmark',   l: 'Benchmark' },
            { k: 'ablation',    l: 'Ablation' },
            { k: 'calibration', l: 'Calibration' },
          ].map(t => (
            <button key={t.k} onClick={() => setTab(t.k as any)}
              className={clsx(
                'px-3 py-1.5 rounded text-[12px] font-medium transition-all',
                tab === t.k ? 'bg-teal/10 text-teal' : 'text-t3 hover:text-t2',
              )}>
              {t.l}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-auto px-5 py-5 space-y-5">

        {/* ── Run controls (benchmark + ablation tabs) ────── */}
        {(tab === 'benchmark' || tab === 'ablation') && (
          <div className="card p-4 flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <label className="t-label">Seeds</label>
              <input className="input w-40 text-[12px] py-1 t-mono" placeholder="42,123,456"
                value={seeds} onChange={e => setSeeds(e.target.value)} />
              <span className="text-t3 text-[11px]">comma-separated</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="t-label">Model</label>
              <div className="flex gap-0.5 bg-ink border border-line2 rounded p-0.5">
                {(['rf', 'gp'] as const).map(m => (
                  <button key={m} onClick={() => setModelType(m)}
                    className={clsx('px-3 py-1 rounded text-[11px] font-medium transition-all',
                      modelType === m ? 'bg-teal/10 text-teal' : 'text-t3 hover:text-t2')}>
                    {m.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-start gap-1.5 text-[11px] text-t3 ml-2 max-w-xs">
              <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-teal" />
              RF is recommended for benchmarks (faster). GP for smaller budgets with higher fidelity.
            </div>
            <button
              className="btn btn-teal text-[12px] ml-auto"
              onClick={() => tab === 'benchmark' ? bench.mutate() : abl.mutate()}
              disabled={bench.isPending || abl.isPending}
            >
              {(bench.isPending || abl.isPending)
                ? <><RefreshCw className="w-3.5 h-3.5 animate-spin" />Running…</>
                : <><Play className="w-3.5 h-3.5" />{tab === 'benchmark' ? 'Run Benchmark' : 'Run Ablation'}</>
              }
            </button>
          </div>
        )}

        {/* ── Errors ───────────────────────────────────────── */}
        {(bench.isError || abl.isError) && (
          <div className="card p-3 border-err/20 text-err text-[12px] flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            {(bench.error || abl.error as any)?.response?.data?.detail ?? 'Run failed.'}
          </div>
        )}

        {/* ── BENCHMARK TAB ──────────────────────────────── */}
        {tab === 'benchmark' && (
          !benchResult ? (
            <EmptyState
              title="No benchmark results yet"
              description="Click 'Run Benchmark' to compare all 4 strategies under matched conditions. Each strategy will use the same initial labeled set and total budget."
            />
          ) : (
            <>
              {/* Summary table */}
              <div className="card overflow-hidden">
                <div className="px-4 py-3 border-b border-line2 flex items-center justify-between">
                  <span className="t-title text-[13px]">Performance Summary</span>
                  <span className="text-t3 text-[11px]">Ranked by NAUDC ↑ · {benchResult.n_seeds} seed(s) · {benchResult.model_type.toUpperCase()} model</span>
                </div>
                <table className="tbl w-full">
                  <thead>
                    <tr>
                      <th>Rank</th><th>Strategy</th>
                      <th className="text-right">NAUDC ↑</th>
                      <th className="text-right">Final MAE ↓</th>
                      <th className="text-right">Final RMSE ↓</th>
                      <th className="text-right">Budget</th>
                      <th className="text-right">Runtime</th>
                      <th className="text-right">Regret ↓</th>
                      <th className="text-right">Top-10 Rate ↑</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchResult.summary_table.map(row => {
                      const m = SM[row.strategy] ?? { label: row.strategy, color: '#888' }
                      return (
                        <tr key={row.strategy} className={row.rank === 1 ? 'bg-[rgba(65,217,140,0.04)]' : ''}>
                          <td>
                            <div className="flex items-center gap-1.5">
                              {row.rank === 1 && <Trophy className="w-3 h-3 text-amber" />}
                              <span className="t-mono font-bold text-t1">{row.rank ? `#${row.rank}` : '—'}</span>
                            </div>
                          </td>
                          <td>
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 rounded-sm" style={{ background: m.color }} />
                              <span className="font-medium text-t1">{m.label}</span>
                              {row.rank === 1 && <span className="chip chip-ok text-[10px]">Best</span>}
                              <span className="text-t3 text-[10px]">
                                {row.n_seeds}/{row.n_requested_seeds} complete
                              </span>
                            </div>
                          </td>
                          <td className="t-mono text-right text-ok font-semibold">
                            {formatMeanStd(row.naudc_mean, row.naudc_std)}
                          </td>
                          <td className="t-mono text-right">{formatMeanStd(row.final_mae_mean, row.final_mae_std)}</td>
                          <td className="t-mono text-right text-err">{formatMeanStd(row.final_rmse_mean, row.final_rmse_std)}</td>
                          <td className="t-mono text-right">{formatMeanStd(row.actual_budget_mean, row.actual_budget_std, 1)}</td>
                          <td className="t-mono text-right">{formatMeanStd(row.runtime_seconds_mean, row.runtime_seconds_std, 1)}s</td>
                          <td className="t-mono text-right text-teal">{formatMeanStd(row.final_regret_mean, row.final_regret_std)}</td>
                          <td className="t-mono text-right">
                            {row.top10_mean == null
                              ? '—'
                              : `${formatMeanStd(row.top10_mean * 100, row.top10_std == null ? null : row.top10_std * 100, 1)}%`}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                <div className="px-4 py-2 border-t border-line2 text-[11px] text-t3">
                  NAUDC = Normalized Area Under Discovery Curve ∈ [0,1]. Higher = discovers better materials earlier.
                  {benchResult.n_seeds > 1 && ' Results averaged across completed seeds; ± values are sample SD.'}
                  {benchResult.n_seeds === 1 && ' One completed seed; variability is not estimable.'}
                </div>
              </div>
              {benchResult.failed_or_incomplete_runs.length > 0 && (
                <div className="card p-3 border-amber/20 text-[12px]">
                  <div className="font-medium text-amber mb-1">Failed or incomplete runs</div>
                  {benchResult.failed_or_incomplete_runs.map(run => (
                    <div key={`${run.strategy}-${run.seed}`} className="text-t3">
                      {SM[run.strategy]?.label ?? run.strategy}, seed {run.seed}: {run.status}
                      {' · '}budget {run.actual_budget}
                      {run.error ? ` · ${run.error}` : ''}
                    </div>
                  ))}
                </div>
              )}

              {/* Charts */}
              <div className="grid grid-cols-2 gap-4">
                <DiscoveryCurveChart series={buildSeries(benchResult, 'discovery_curve')}
                  title="Best-Found Property vs Budget" yKey="values" />
                <DiscoveryCurveChart series={buildSeries(benchResult, 'rmse_curve')}
                  title="Test RMSE vs Training Budget" yKey="values" />
                <DiscoveryCurveChart series={buildSeries(benchResult, 'regret_curve')}
                  title="Cumulative Regret vs Budget" yKey="values" />
                <DiscoveryCurveChart series={buildSeries(benchResult, 'diversity_curve')}
                  title="Batch Diversity vs Budget" yKey="values" />
              </div>

              {/* Config note */}
              <div className="card p-4 text-[11px] text-t3">
                <div className="t-label mb-1.5">Benchmark Configuration</div>
                <div className="grid grid-cols-2 gap-x-8 gap-y-1">
                  {[
                    ['Initial samples', benchResult.configuration.initial_samples],
                    ['Batch size', benchResult.configuration.batch_size],
                    ['Total budget', benchResult.configuration.total_budget],
                    ['Base seed', benchResult.configuration.base_seed],
                    ['Seeds used', benchResult.seeds.join(', ')],
                    ['Model', benchResult.model_type.toUpperCase()],
                    ['Runtime', `${benchResult.elapsed_seconds}s`],
                    ['Benchmark ID', benchResult.benchmark_id],
                    ['Status', benchResult.benchmark_status],
                    ['Dataset fingerprint', benchResult.configuration.dataset_fingerprint.slice(0, 12)],
                  ].map(([k, v]) => (
                    <div key={String(k)}><span className="text-t2 font-medium">{k}: </span><span className="t-mono">{v}</span></div>
                  ))}
                </div>
                <div className="mt-2 text-t4">
                  Strategies share the same initial set within each seed. The fixed test split is used only
                  for prediction metrics, not acquisition or strategy ranking. Discovery reference values
                  are computed from the training pool.
                </div>
              </div>
            </>
          )
        )}

        {/* ── ABLATION TAB ───────────────────────────────── */}
        {tab === 'ablation' && (
          !ablResult ? (
            <EmptyState
              title="No ablation results yet"
              description="Click 'Run Ablation' to compare 4 PUDA variants: prediction-only, pred+uncertainty, pred+diversity, and full PUDA. This tests whether each component contributes."
            />
          ) : (
            <>
              <div className="card p-4">
                <div className="t-title text-[13px] mb-3">PUDA Component Ablation</div>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  {Object.entries(ablResult.formulas ?? {}).map(([variant, formula]) => {
                    const m = SM[variant] ?? { label: variant, color: '#888' }
                    const row = ablResult.summary_table.find(r => r.strategy === variant)
                    return (
                      <div key={variant} className="card p-3">
                        <div className="flex items-center gap-2 mb-1.5">
                          <div className="w-2 h-2 rounded-sm" style={{ background: m.color }} />
                          <span className="text-[12px] font-medium text-t1">{m.label}</span>
                          {row?.rank === 1 && <span className="chip chip-ok text-[10px]">Best</span>}
                        </div>
                        <div className="t-mono text-[10px] text-t3 mb-2">{formula}</div>
                        {row && (
                          <div className="flex gap-3 text-[11px]">
                            <span className="text-t3">NAUDC <span className="t-mono text-ok">{row.naudc_mean?.toFixed(4) ?? '—'}</span></span>
                            <span className="text-t3">Regret <span className="t-mono text-teal">{row.final_regret_mean?.toFixed(4) ?? '—'}</span></span>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
                {ablResult.interpretation && (
                  <div className="inset p-3 text-[11px] text-t3 leading-relaxed">
                    <span className="text-t2 font-medium">Interpretation: </span>
                    {ablResult.interpretation}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <DiscoveryCurveChart series={buildSeries(ablResult, 'discovery_curve')}
                  title="Best-Found: PUDA Ablation" yKey="values" />
                <DiscoveryCurveChart series={buildSeries(ablResult, 'rmse_curve')}
                  title="RMSE: PUDA Ablation" yKey="values" />
              </div>
            </>
          )
        )}

        {/* ── CALIBRATION TAB ────────────────────────────── */}
        {tab === 'calibration' && (
          calLoading ? (
            <div className="text-t3 text-[12px] text-center py-8 flex items-center justify-center gap-2">
              <RefreshCw className="w-4 h-4 animate-spin" />Loading calibration metrics…
            </div>
          ) : !cal || 'error' in (cal as any) ? (
            <EmptyState
              title="Insufficient calibration data"
              description={`${(cal as any)?.error ?? 'Run an iteration to compute calibration metrics.'} Valid pairs: ${(cal as any)?.n_aligned ?? 0}; excluded: ${(cal as any)?.n_excluded ?? 0}.`}
            />
          ) : (
            <>
              <div className="text-t3 text-[11px]">
                Calibration RMSE uses pre-acquisition predictions for acquired candidates; it is distinct from held-out test RMSE. The sample is acquisition-selected, and small-sample coverage is noisy, not proof of calibration.
              </div>
              {/* Key metrics */}
              <div className="grid grid-cols-4 gap-4">
                {[
                  { l: 'ECE',           v: (cal as any).ece?.toFixed(4),              hint: 'Lower = better calibrated', c: (cal as any).ece < 0.1 ? '#41D98C' : '#F06565' },
                  { l: 'Coverage 90%',  v: `${((cal as any).coverage_90pct*100).toFixed(1)}%`, hint: 'Should be near 90%', c: Math.abs((cal as any).coverage_90pct - 0.90) < 0.05 ? '#41D98C' : '#EDAB60' },
                  { l: 'Coverage 95%',  v: `${((cal as any).coverage_95pct*100).toFixed(1)}%`, hint: 'Should be near 95%', c: Math.abs((cal as any).coverage_95pct - 0.95) < 0.05 ? '#41D98C' : '#EDAB60' },
                  { l: 'RMSE (pre-acq.)', v: (cal as any).rmse_on_observed?.toFixed(4), hint: `${(cal as any).n_aligned} valid pairs`, c: '#A99BFF' },
                ].map(({ l, v, hint, c }) => (
                  <div key={l} className="card p-3.5">
                    <div className="t-label mb-1">{l}</div>
                    <div className="t-mono font-bold text-lg" style={{ color: c }}>{v}</div>
                    <div className="text-t3 text-[11px] mt-0.5">{hint}</div>
                  </div>
                ))}
              </div>

              {/* Reliability diagram */}
              {(cal as any).reliability_diagram && (
                <div className="card p-4">
                  <div className="t-label mb-3">Reliability Diagram</div>
                  <div className="overflow-x-auto">
                    <table className="tbl">
                      <thead><tr><th>Expected Coverage</th><th className="text-right">Observed Coverage</th><th className="text-right">Δ</th></tr></thead>
                      <tbody>
                        {(cal as any).reliability_diagram.expected_coverage.map((exp: number, i: number) => {
                          const obs = (cal as any).reliability_diagram.observed_coverage[i]
                          const diff = obs - exp
                          return (
                            <tr key={i}>
                              <td className="t-mono">{(exp*100).toFixed(0)}%</td>
                              <td className="t-mono text-right">{(obs*100).toFixed(1)}%</td>
                              <td className={clsx('t-mono text-right', Math.abs(diff) < 0.05 ? 'text-ok' : 'text-err')}>
                                {diff >= 0 ? '+' : ''}{(diff*100).toFixed(1)}%
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              <div className="text-t3 text-[11px]">
                n={(cal as any).n_observations}; excluded={(cal as any).n_excluded}. Coverage ECE summarizes interval coverage gaps; it does not assess the full predictive distribution.
              </div>

              {/* Notes */}
              <div className="card p-4 text-[11px] text-t3 space-y-1.5">
                <div><span className="text-t2 font-medium">Interpretation: </span>{(cal as any).interpretation}</div>
                <div className="text-t4"><span className="text-amber font-medium">Caveat: </span>{(cal as any).caveat}</div>
              </div>
            </>
          )
        )}

        {/* Methodology note always visible */}
        <div className="card p-4 text-[11px] text-t3">
          <div className="t-label mb-1.5">Benchmark Methodology</div>
          <div className="grid grid-cols-2 gap-x-8 gap-y-1">
            {[
              ['Test set', '20% held out before AL loop — never seen during acquisition'],
              ['NAUDC', '∫ best_found/y_max dt/T — budget efficiency scalar ∈ [0,1]'],
              ['Regret', 'global_max − best_found(n) — should decrease each iteration'],
              ['Diversity', 'Mean pairwise cosine distance in selected batch'],
              ['Evaluation', 'Retrospective Simulated Evaluation — true values from held-out dataset'],
              ['Reproducibility', 'All runs use same initial labeled set from matched random seed'],
            ].map(([k, v]) => <div key={String(k)}><span className="text-t2 font-medium">{k}: </span>{v}</div>)}
          </div>
        </div>
      </div>
    </div>
  )
}
