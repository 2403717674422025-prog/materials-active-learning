import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Play, RefreshCw, ChevronRight, Award } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { api } from '../services/api'
import type { ExperimentStatusResponse, IterationResult, CandidatePrediction } from '../types'
import BudgetMeter from '../components/BudgetMeter'
import StatusBadge from '../components/StatusBadge'
import UncertaintyBadge from '../components/UncertaintyBadge'
import { SkCard, SkTable } from '../components/Skeleton'
import clsx from 'clsx'

const CT = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card p-2 text-[11px] min-w-[140px]">
      <div className="t-label mb-1.5">Iter {label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex justify-between gap-3">
          <span className="text-t3">{p.name}</span>
          <span className="t-mono" style={{ color: p.color }}>{p.value?.toFixed(4)}</span>
        </div>
      ))}
    </div>
  )
}

export default function ActiveLearningDashboardPage() {
  const { id } = useParams<{ id: string }>()
  const qc = useQueryClient()
  const [selectedCand, setSelectedCand] = useState<CandidatePrediction | null>(null)
  const [candPage, setCandPage] = useState(1)

  const { data: st, isLoading } = useQuery<ExperimentStatusResponse>({
    queryKey: ['experiment-status', id],
    queryFn: () => api.experiments.status(id!),
    refetchInterval: 6_000,
  })

  const { data: history } = useQuery({
    queryKey: ['experiment-history', id],
    queryFn: () => api.experiments.history(id!),
    enabled: !!id,
  })

  const { data: tuningRuns } = useQuery({
    queryKey: ['experiment-tuning-runs', id],
    queryFn: () => api.experiments.tuningRuns(id!),
    enabled: !!id,
  })

  const { data: cands, isLoading: cl } = useQuery({
    queryKey: ['experiment-candidates', id, candPage],
    queryFn: () => api.experiments.candidates(id!, undefined, candPage, 80),
    enabled: !!id,
  })

  const run = useMutation<IterationResult>({
    mutationFn: () => api.experiments.runIteration(id!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['experiment-status', id] })
      qc.invalidateQueries({ queryKey: ['experiment-history', id] })
      qc.invalidateQueries({ queryKey: ['experiment-candidates', id] })
    },
  })

  const chart = history?.map(h => ({
    i: h.iteration,
    rmse: +(h.test_rmse?.toFixed(4) ?? 0),
    best: +(h.best_found_value?.toFixed(4) ?? 0),
    regret: +(h.regret?.toFixed(4) ?? 0),
  })) ?? []

  const canRun = st && ['initialized','running'].includes(st.status) && !run.isPending

  if (isLoading) return (
    <div className="p-6 grid grid-cols-4 gap-4">{[0,1,2,3].map(i => <SkCard key={i} />)}</div>
  )
  if (!st) return <div className="p-6 text-t3 text-[12px]">Experiment not found.</div>

  return (
    <div className="flex h-full">
      {/* ── Main: table + charts ──────────────────────── */}
      <div className="flex-1 min-w-0 flex flex-col overflow-hidden">

        {/* Console header */}
        <div className="flex items-center gap-4 px-5 py-3 border-b border-line2 bg-ink1">
          <div className="flex items-center gap-2.5">
            <span className="dot dot-live" />
            <span className="t-mono text-[12px] text-t2">{id?.slice(0,8)}…</span>
          </div>
          <div className="h-3 border-r border-line2" />
          <StatusBadge status={st.status} />
          <span className="text-t3 text-[12px]">Iteration <span className="text-t1 font-semibold">{st.current_iteration}</span></span>
          <div className="flex-1" />
          {run.isError && (
            <span className="text-err text-[11px]">
              {(run.error as any)?.response?.data?.detail ?? 'Iteration failed'}
            </span>
          )}
          <button className="btn btn-teal text-[12px] py-1.5" onClick={() => run.mutate()} disabled={!canRun}>
            {run.isPending ? <><RefreshCw className="w-3.5 h-3.5 animate-spin" />Running…</>
                           : <><Play className="w-3.5 h-3.5" />Run Iteration</>}
          </button>
        </div>

        {/* Stat strip */}
        <div className="flex items-stretch border-b border-line2">
          {[
            { l: 'RMSE',        v: st.latest_rmse?.toFixed(4) ?? '—',        c: '#A99BFF', hint: 'test set' },
            { l: 'R²',          v: st.latest_r2?.toFixed(4) ?? '—',           c: '#41D98C', hint: 'test set' },
            { l: 'Regret',      v: st.latest_regret?.toFixed(4) ?? '—',       c: '#59D5CC', hint: 'eV from global max' },
            { l: 'Best Found',  v: st.best_found_value?.toFixed(4) ?? '—',    c: '#EDAB60', hint: st.best_found_id?.slice(0,14) ?? '' },
            { l: 'Global Max',  v: st.global_maximum?.toFixed(4) ?? '—',      c: '#9AA9BA', hint: 'eV' },
          ].map(({ l, v, c, hint }) => (
            <div key={l} className="stat-item py-3 first:pl-5">
              <span className="t-label">{l}</span>
              <span className="stat-val" style={{ color: c }}>{v}</span>
              {hint && <span className="t-label text-[10px] mt-0.5">{hint}</span>}
            </div>
          ))}
        </div>

        {tuningRuns?.[0] && (
          <section className="border-b border-line2 bg-ink1/60 px-5 py-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="t-label">Hyperparameter optimization · {tuningRuns[0].status}</div>
                <div className="text-[11px] text-t3 mt-1">
                  Validation RMSE: <span className="t-mono text-teal">
                    {tuningRuns[0].best_validation_rmse?.toFixed(4) ?? '—'}
                  </span>
                  {' · '}training/validation: {tuningRuns[0].training_size ?? '—'}/{tuningRuns[0].validation_size ?? '—'}
                  {' · '}seed {tuningRuns[0].random_seed}
                </div>
                {tuningRuns[0].best_params && (
                  <div className="t-mono text-[10px] text-t2 mt-1 break-all">
                    Best parameters: {JSON.stringify(tuningRuns[0].best_params)}
                  </div>
                )}
              </div>
              <div className="text-[10px] text-t4">
                Validation score is separate from the held-out test RMSE above.
              </div>
            </div>
            {tuningRuns[0].trial_history.length > 0 && (
              <details className="mt-2">
                <summary className="cursor-pointer text-[10px] text-teal">
                  Trial history ({tuningRuns[0].completed_trials} complete, {tuningRuns[0].failed_trials} failed)
                </summary>
                <div className="overflow-auto mt-2 max-h-36">
                  <table className="tbl w-full text-[10px]">
                    <thead><tr><th>Trial</th><th>Status</th><th className="text-right">Validation RMSE</th><th>Parameters</th></tr></thead>
                    <tbody>
                      {tuningRuns[0].trial_history.map(trial => (
                        <tr key={trial.number}>
                          <td className="t-mono">{trial.number}</td>
                          <td>{trial.state}</td>
                          <td className="t-mono text-right">{trial.value?.toFixed(5) ?? '—'}</td>
                          <td className="t-mono text-t4">{JSON.stringify(trial.params)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
          </section>
        )}

        {/* Candidates table */}
        <div className="flex-1 overflow-auto">
          {cl ? <SkTable rows={10} /> : (
            <table className="tbl w-full">
              <thead>
                <tr>
                  <th>Candidate ID</th>
                  <th className="text-right">Predicted (eV)</th>
                  <th className="text-right">95% interval (GPR)</th>
                  <th className="text-center">Uncertainty</th>
                  <th className="text-right">Acq. Score</th>
                  <th className="text-center">Status</th>
                  <th className="text-center">Rank</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {cands?.candidates.map(c => (
                  <tr key={c.candidate_id}
                    className={clsx(
                      'cursor-pointer',
                      c.is_selected && 'bg-[rgba(237,171,96,0.04)]',
                      selectedCand?.candidate_id === c.candidate_id && 'bg-[rgba(89,213,204,0.06)]',
                    )}
                    onClick={() => setSelectedCand(prev => prev?.candidate_id === c.candidate_id ? null : c)}
                  >
                    <td className="t-mono text-[11px] text-t1 font-medium">{c.candidate_id}</td>
                    <td className="t-mono text-right font-semibold text-t1">{c.predicted_mean.toFixed(4)}</td>
                    <td className="t-mono text-right text-t3 text-[11px]">{c.lower_cb == null || c.upper_cb == null ? 'Not calibrated' : `[${c.lower_cb.toFixed(2)},${c.upper_cb.toFixed(2)}]`}</td>
                    <td className="text-center"><UncertaintyBadge label={c.uncertainty_label} value={c.predicted_std} showValue /></td>
                    <td className="t-mono text-right font-semibold" style={{ color: '#59D5CC' }}>{c.acquisition_score.toFixed(4)}</td>
                    <td className="text-center">
                      {c.is_selected ? <span className="chip chip-amber">Selected</span> : <span className="text-t4">—</span>}
                    </td>
                    <td className="t-mono text-center text-t3 text-[11px]">{c.selection_rank ?? '—'}</td>
                    <td>
                      <Link to={`/experiment/${id}/candidates/${encodeURIComponent(c.candidate_id)}`}
                        className="text-t4 hover:text-teal transition-colors" onClick={e => e.stopPropagation()}>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {cands && cands.total_pages > 1 && (
            <div className="flex items-center justify-between px-4 py-2.5 border-t border-line2 text-[12px]">
              <span className="text-t3">Page {candPage}/{cands.total_pages} · {cands.total_candidates.toLocaleString()} total</span>
              <div className="flex gap-2">
                <button className="btn btn-ghost text-[11px] py-1" onClick={() => setCandPage(p => Math.max(1,p-1))} disabled={candPage===1}>Prev</button>
                <button className="btn btn-ghost text-[11px] py-1" onClick={() => setCandPage(p => Math.min(cands.total_pages,p+1))} disabled={candPage===cands.total_pages}>Next</button>
              </div>
            </div>
          )}
        </div>

        {/* Charts */}
        {chart.length >= 2 && (
          <div className="border-t border-line2 grid grid-cols-2 divide-x divide-line2 flex-shrink-0">
            {[
              { key: 'rmse',   label: 'RMSE',        color: '#A99BFF', unit: 'eV' },
              { key: 'best',   label: 'Best Found',  color: '#59D5CC', unit: 'eV' },
            ].map(({ key, label, color, unit }) => (
              <div key={key} className="p-4">
                <div className="t-label mb-2">{label}</div>
                <ResponsiveContainer width="100%" height={90}>
                  <LineChart data={chart} margin={{ top: 2, right: 8, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="2 6" stroke="rgba(39,51,67,0.5)" vertical={false} />
                    <XAxis dataKey="i" tick={{ fill: '#5C6E82', fontSize: 9 }} />
                    <YAxis tick={{ fill: '#5C6E82', fontSize: 9 }} />
                    <Tooltip content={<CT />} />
                    <Line type="monotone" dataKey={key} stroke={color} strokeWidth={1.5} dot={false} name={`${label} (${unit})`} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Right: candidate detail drawer ──────────────── */}
      <div className={clsx(
        'border-l border-line2 bg-ink1 transition-all flex-shrink-0 overflow-hidden',
        selectedCand ? 'w-[260px]' : 'w-0',
      )}>
        {selectedCand && <CandDrawer cand={selectedCand} expId={id!} onClose={() => setSelectedCand(null)} />}
      </div>
    </div>
  )
}

function CandDrawer({ cand, expId, onClose }: { cand: CandidatePrediction; expId: string; onClose: () => void }) {
  const { data: detail } = useQuery({
    queryKey: ['candidate-detail', expId, cand.candidate_id],
    queryFn: () => api.experiments.candidateDetail(expId, cand.candidate_id),
  })
  const bd = (detail?.full_breakdown ?? {}) as Record<string, number>

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b border-line2">
        <div className="t-section">Candidate Detail</div>
        <button onClick={onClose} className="text-t4 hover:text-t2 text-[18px] leading-none">×</button>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        <div className="t-mono text-[12px] text-t1 font-semibold break-all">{cand.candidate_id}</div>

        <div className="space-y-2 text-[12px]">
          {[
            { l: 'Predicted',   v: `${cand.predicted_mean.toFixed(4)} eV`, c: '#F3F5F8' },
            { l: 'Uncertainty', v: `±${cand.predicted_std.toFixed(4)} eV`, c: '#A99BFF' },
            { l: '95% interval (GPR)', v: cand.lower_cb == null || cand.upper_cb == null ? 'Not calibrated' : `[${cand.lower_cb.toFixed(3)},${cand.upper_cb.toFixed(3)}]`, c: '#9AA9BA' },
            { l: 'Acq. Score', v: cand.acquisition_score.toFixed(4), c: '#59D5CC' },
          ].map(({ l, v, c }) => (
            <div key={l} className="flex items-center justify-between gap-2">
              <span className="text-t3">{l}</span>
              <span className="t-mono font-medium" style={{ color: c }}>{v}</span>
            </div>
          ))}
        </div>

        <hr className="rule2" />

        {/* Acquisition breakdown */}
        {bd.alpha !== undefined && (
          <div className="space-y-2">
            <div className="t-label">Acquisition Breakdown</div>
            {[
              { l: 'Performance', v: (bd.alpha ?? 0) * (bd.prediction_normalized ?? 0), c: '#59D5CC' },
              { l: 'Uncertainty', v: (bd.beta ?? 0)  * (bd.uncertainty_normalized ?? 0), c: '#A99BFF' },
              { l: 'Diversity',   v: (bd.gamma ?? 0) * (bd.diversity_normalized ?? 0),   c: '#EDAB60' },
            ].map(({ l, v, c }) => (
              <div key={l} className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-sm flex-shrink-0" style={{ background: c }} />
                <span className="text-t3 text-[11px] w-20">{l}</span>
                <div className="flex-1 h-1 bg-panel3 rounded-sm overflow-hidden">
                  <div style={{ width:`${v*100}%`, background: c, opacity: 0.7 }} className="h-full" />
                </div>
                <span className="t-mono text-[11px] text-t2 w-12 text-right">{v.toFixed(4)}</span>
              </div>
            ))}
          </div>
        )}

        {detail?.selection_reason && (
          <>
            <hr className="rule2" />
            <div>
              <div className="t-label mb-1.5">Selection Reason</div>
              <p className="text-t3 text-[11px] leading-relaxed">{detail.selection_reason}</p>
            </div>
          </>
        )}

        <Link to={`/experiment/${expId}/candidates/${encodeURIComponent(cand.candidate_id)}`}
          className="btn btn-ghost w-full text-[12px] justify-center mt-2">
          Full Breakdown →
        </Link>
      </div>
    </div>
  )
}
