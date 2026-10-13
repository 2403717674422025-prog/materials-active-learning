import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  FlaskConical, Database, ArrowRight, RefreshCw,
  TrendingUp, AlertTriangle, Play, Info, ChevronRight,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ScatterChart, Scatter,
  ZAxis, BarChart, Bar, ReferenceLine, Legend,
} from 'recharts'
import { api } from '../services/api'
import type { ExperimentStatusResponse, EmbeddingPoint } from '../types'
import clsx from 'clsx'

// ── Color helpers for scatter ─────────────────────────────────────────────────
function ptColor(pt: EmbeddingPoint, mode: string): string {
  if (mode === 'observation') return pt.is_observed ? '#20C7B7' : '#26394C'
  if (mode === 'selected')    return pt.is_selected  ? '#F6B84A' : pt.is_observed ? '#20C7B7' : '#26394C'

  const val = mode === 'uncertainty' ? pt.predicted_std
            : mode === 'score'       ? pt.acquisition_score
            : pt.predicted_mean

  if (val == null) return '#26394C'
  const t = 0.5 + 0.5 * Math.tanh(val)
  const r = Math.round(32 + (167 - 32) * t)
  const g = Math.round(220 + (139 - 220) * t)
  const b = Math.round(232 + (250 - 232) * t)
  return `rgb(${r},${g},${b})`
}

const CustomDot = (props: any) => {
  const { cx, cy, payload, colorMode } = props
  const c = ptColor(payload, colorMode)
  const r = payload.is_selected ? 5 : payload.is_observed ? 4 : 3
  return (
    <circle cx={cx} cy={cy} r={r}
      fill={c} opacity={payload.is_observed || payload.is_selected ? 0.95 : 0.5}
      stroke={payload.is_selected ? '#F6B84A' : 'none'} strokeWidth={payload.is_selected ? 1.5 : 0} />
  )
}

const CT = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload as EmbeddingPoint
  return (
    <div className="card p-2.5 text-[11px] min-w-[190px]">
      <div className="t-mono text-t1 font-semibold truncate mb-1.5">{d.candidate_id}</div>
      <div className="space-y-0.5 text-t2">
        <div className="flex justify-between gap-4"><span className="text-t3">Status</span>
          <span className={d.is_observed ? 'text-amber' : 'text-t3'}>{d.is_observed ? 'Observed' : 'Unobserved'}</span></div>
        {d.true_value != null && <div className="flex justify-between gap-4"><span className="text-t3">True value</span><span className="t-mono">{d.true_value.toFixed(4)} eV</span></div>}
        {d.predicted_mean != null && <div className="flex justify-between gap-4"><span className="text-t3">Predicted</span><span className="t-mono">{d.predicted_mean.toFixed(4)} eV</span></div>}
        {d.predicted_std != null && <div className="flex justify-between gap-4"><span className="text-t3">Uncertainty</span><span className="t-mono">{d.predicted_std.toFixed(4)} eV</span></div>}
        {d.acquisition_score != null && <div className="flex justify-between gap-4"><span className="text-t3">Acq. score</span><span className="t-mono">{d.acquisition_score.toFixed(4)}</span></div>}
        {d.is_selected && <div className="text-amber mt-1 font-medium">● Selected this iteration</div>}
      </div>
    </div>
  )
}

// ── Stat strip ────────────────────────────────────────────────────────────────
function StatStrip({ ds, st, planned }: { ds: any; st?: ExperimentStatusResponse; planned?: number }) {
  const items = [
    { label: 'Total Candidates', val: ds?.num_samples?.toLocaleString() ?? '—', unit: ds?.name ?? 'materials', color: '#20DCE8' },
    { label: 'Observed', val: st?.budget_used ?? '—', unit: st && ds?.num_samples ? `${((st.budget_used / ds.num_samples) * 100).toFixed(1)}% of pool` : 'labeled', color: '#E8F0FA' },
    { label: 'Budget Used', val: st ? `${st.budget_used}/${st.budget_total}` : '—', unit: st ? `${st.budget_remaining} remaining` : 'experiments', color: '#F6B84A' },
    { label: 'Best Observed', val: st?.best_found_value?.toFixed(3) ?? '—', unit: 'eV · measured', color: '#20C7B7' },
    { label: 'Iteration', val: st?.current_iteration ?? '—', unit: st ? `of ${planned ?? '—'}` : 'planned', color: '#A78BFA' },
  ]
  return (
    <div className="flex items-stretch border-b border-line2 bg-ink1 text-sm">
      {items.map((it, i) => (
        <div key={it.label} className={clsx(
          'stat-item py-3',
          i === 0 && 'pl-5',
          i === items.length - 1 && 'border-r-0 pr-5',
        )} style={{ flex: '1 1 0', minWidth: 0 }}>
          <span className="t-label block mb-0.5">{it.label}</span>
          <span className="stat-val" style={{ color: it.color }}>
            {it.val}
            {it.unit && <span className="text-t3 font-normal text-xs ml-1">{it.unit}</span>}
          </span>
          {it.label === 'Budget Used' && st && <div className="mt-1.5 h-1 bg-panel3 overflow-hidden"><div className="h-full bg-amber" style={{ width: `${Math.min(100, st.budget_pct)}%` }} /></div>}
        </div>
      ))}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function OverviewPage() {
  const qc = useQueryClient()
  const [colorMode, setColorMode] = useState<'prediction' | 'uncertainty' | 'score' | 'observation' | 'selected'>('prediction')
  const [activeExpId, setActiveExpId] = useState<string | null>(null)
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null)
  const [idInput, setIdInput] = useState('')

  const { data: datasets, isLoading: datasetsLoading, isError: datasetsError, error: datasetError } = useQuery({ queryKey: ['datasets'], queryFn: api.datasets.list })
  const { data: expList, isLoading: experimentsLoading, isError: experimentsError } = useQuery({
    queryKey: ['experiments-list'],
    queryFn: () => api.experiments.list(0, 10),
  })
  useEffect(() => {
    if (!activeExpId && expList?.length) setActiveExpId(expList[0].id)
  }, [activeExpId, expList])
  const activeExperiment = expList?.find(e => e.id === activeExpId)
  const ds = datasets?.find(d => d.id === activeExperiment?.dataset_id) ?? datasets?.find(d => d.status === 'ready')
  const datasetUsable = !!ds && ds.status === 'ready' && (ds.num_samples ?? 0) > 0 && (ds.num_features ?? 0) > 0

  const { data: st } = useQuery<ExperimentStatusResponse>({
    queryKey: ['experiment-status', activeExpId],
    queryFn: () => api.experiments.status(activeExpId!),
    enabled: !!activeExpId, refetchInterval: 8_000,
  })

  const { data: embedding, isLoading: embLoading, error: embError } = useQuery({
    queryKey: ['embedding', activeExpId, colorMode],
    queryFn: () => api.experiments.embedding(activeExpId!),
    enabled: !!activeExpId && !!st && !!ds && ds.status === 'ready' && (ds.num_samples ?? 0) > 0 && (ds.num_features ?? 0) > 0,
    staleTime: 30_000,
  })

  const { data: history } = useQuery({
    queryKey: ['overview-history', activeExpId],
    queryFn: () => api.experiments.history(activeExpId!),
    enabled: !!activeExpId,
  })

  const { data: recommendations } = useQuery({
    queryKey: ['overview-candidates', activeExpId, st?.current_iteration],
    queryFn: () => api.experiments.candidates(activeExpId!, undefined, 1, 500),
    enabled: !!activeExpId && !!st?.current_iteration,
  })
  const { data: calibration } = useQuery({
    queryKey: ['overview-calibration', activeExpId],
    queryFn: () => api.experiments.calibration(activeExpId!),
    enabled: !!activeExpId && !!st?.current_iteration,
  })
  const { data: pointDetail } = useQuery({
    queryKey: ['candidate-detail', activeExpId, selectedPointId],
    queryFn: () => api.experiments.candidateDetail(activeExpId!, selectedPointId!),
    enabled: !!activeExpId && !!selectedPointId && !!st?.current_iteration,
  })
  const { data: discovered } = useQuery({
    queryKey: ['overview-discovery', activeExpId, st?.current_iteration],
    queryFn: () => api.experiments.discovery(activeExpId!, 500),
    enabled: !!activeExpId && !!st?.current_iteration,
  })

  const loadDs = useMutation({
    mutationFn: () => api.datasets.loadBuiltin('matbench_expt_gap', 'gap expt'),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] })
      qc.invalidateQueries({ queryKey: ['experiments-list'] })
    },
  })

  const initialize = useMutation({
    mutationFn: () => api.experiments.initialize(activeExpId!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['experiment-status', activeExpId] })
      qc.invalidateQueries({ queryKey: ['experiments-list'] })
      qc.invalidateQueries({ queryKey: ['embedding', activeExpId] })
    },
  })

  const runIter = useMutation({
    mutationFn: () => api.experiments.runIteration(activeExpId!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['experiment-status', activeExpId] })
      qc.invalidateQueries({ queryKey: ['overview-history', activeExpId] })
      qc.invalidateQueries({ queryKey: ['overview-candidates', activeExpId] })
      qc.invalidateQueries({ queryKey: ['embedding', activeExpId] })
      qc.invalidateQueries({ queryKey: ['overview-calibration', activeExpId] })
      qc.invalidateQueries({ queryKey: ['candidate-detail', activeExpId] })
      qc.invalidateQueries({ queryKey: ['overview-discovery', activeExpId] })
    },
  })

  function selectExperiment(id: string) {
    if (id !== activeExpId) {
      runIter.reset()
      initialize.reset()
      setSelectedPointId(null)
      setActiveExpId(id)
    }
  }

  const chartData = history?.map(h => ({
    i: h.iteration, rmse: h.test_rmse == null ? null : +h.test_rmse.toFixed(4),
    best: h.best_found_value == null ? null : +h.best_found_value.toFixed(4), budget: h.budget_used,
  })) ?? []
  const lastHistory = history?.[history.length - 1]

  const uncertaintyBins = (() => {
    const values = (recommendations?.candidates ?? []).map(c => c.predicted_std).filter(Number.isFinite)
    if (!values.length) return []
    const max = Math.max(...values)
    const width = max > 0 ? max / 8 : 1
    return Array.from({ length: 8 }, (_, i) => ({
      bin: `${(i * width).toFixed(2)}–${((i + 1) * width).toFixed(2)}`,
      count: values.filter(v => (i === 7 ? v >= i * width : v >= i * width && v < (i + 1) * width)).length,
    }))
  })()
  const reliability = calibration?.reliability_diagram
  const calibrationPlot = reliability?.expected_coverage?.map((nominal: number, i: number) => ({
    nominal, ideal: nominal, observed: reliability.observed_coverage[i],
  })) ?? []
  const regressionPoints = (discovered ?? []).filter((m: any) => m.predicted_mean != null).map((m: any) => ({
    actual: m.true_value, predicted: m.predicted_mean, error: m.predicted_mean - m.true_value,
    candidate: m.candidate_id, iteration: m.iteration_discovered,
  }))
  const regressionMae = regressionPoints.length ? regressionPoints.reduce((s: number, p: any) => s + Math.abs(p.error), 0) / regressionPoints.length : null
  const regressionRmse = regressionPoints.length ? Math.sqrt(regressionPoints.reduce((s: number, p: any) => s + p.error * p.error, 0) / regressionPoints.length) : null

  const canRun = st && ['initialized', 'running'].includes(st.status) && !runIter.isPending

  // Color mode label map
  const colorLabels: Record<string, string> = {
    prediction: 'Predicted', uncertainty: 'Uncertainty',
    score: 'Acq. Score', observation: 'Observed', selected: 'Selected',
  }

  return (
    <div className="flex flex-col h-full">
      {/* ── Application header ─────────────────────────────── */}
      <div className="px-5 pt-5 pb-3 border-b border-line2 flex items-start justify-between gap-6">
        <div>
          <div className="text-t3 text-[11px] t-mono mb-1">MATERIALS AL / DISCOVERY INTELLIGENCE</div>
          <h1 style={{ fontSize: '1.2rem', fontWeight: 700, letterSpacing: '-0.025em', color: '#F3F5F8', lineHeight: 1.2 }}>
            Discovery Intelligence
          </h1>
          <p className="text-t3 text-[12px] mt-1 max-w-xl leading-relaxed">
            Explore material candidates using uncertainty-aware Bayesian active learning.
            {activeExperiment && <span className="ml-2 text-teal">{activeExperiment.name || activeExperiment.strategy.toUpperCase()} · {ds?.name ?? 'Dataset unavailable'} · {activeExperiment.strategy.toUpperCase()}</span>}
          </p>
        </div>
        <div className="flex-shrink-0 flex flex-col items-end gap-2 pt-1">
          {!datasetUsable ? (
            <button className="btn btn-panel text-[12px]" onClick={() => loadDs.mutate()} disabled={loadDs.isPending}>
              {loadDs.isPending ? <><RefreshCw className="w-3.5 h-3.5 animate-spin" />Preparing dataset…</> : <><Database className="w-3.5 h-3.5" />{ds ? 'Repair dataset' : 'Load expt_gap'}</>}
            </button>
          ) : (
            <div className="chip chip-ok text-[11px]"><span className="dot dot-ok" />{ds.num_samples?.toLocaleString()} candidates ready</div>
          )}
          <Link to="/setup" className="btn btn-teal text-[12px]">
            <FlaskConical className="w-3.5 h-3.5" />New Experiment
          </Link>
        </div>
      </div>

      {/* ── Stat strip ─────────────────────────────────────── */}
      <StatStrip ds={ds} st={st ?? undefined} planned={activeExperiment ? Math.ceil(activeExperiment.total_budget / Math.max(1, activeExperiment.batch_size)) : undefined} />

      {/* ── Main workspace ─────────────────────────────────── */}
      <div className="flex-1 min-h-0 flex overflow-hidden">

        {/* Center: scatter + timeline ──────────────────────── */}
        <div className="flex-1 min-w-0 flex flex-col overflow-hidden">

          {/* Viz header */}
          <div className="flex items-center gap-3 px-5 py-2.5 border-b border-line2 flex-wrap">
            <div className="t-title text-[13px]">Material Search Space</div>
            {embedding && (
                <span className="text-t3 text-[11px]">
                {embedding.method} on standardized features · {embedding.n_shown.toLocaleString()} points ·{' '}
                {(embedding.total_explained * 100).toFixed(1)}% variance explained ·{' '}
                <span className="text-amber">{embedding.n_observed} observed</span>
              </span>
            )}
            {embedding && (
              <div className="flex items-center gap-2 text-[9px] text-t3">
                {colorMode === 'observation' ? <><span className="w-2 h-2 rounded-full bg-teal" />Observed<span className="w-2 h-2 rounded-full bg-[#26394C] ml-1" />Unobserved</> : colorMode === 'selected' ? <><span className="w-2 h-2 rounded-full bg-amber" />Selected<span className="w-2 h-2 rounded-full bg-teal ml-1" />Observed<span className="w-2 h-2 rounded-full bg-[#26394C] ml-1" />Unobserved</> : <><span>Low</span><span className="w-12 h-1.5 rounded-sm" style={{ background: 'linear-gradient(90deg,#20DCE8,#A78BFA)' }} /><span>High</span></>}
              </div>
            )}
            <div className="flex-1" />

            {/* Color mode switcher */}
            <div className="flex items-center gap-0.5 bg-ink1 border border-line2 rounded p-0.5">
              {Object.entries(colorLabels).map(([m, label]) => (
                <button key={m} onClick={() => setColorMode(m as any)}
                  className={clsx(
                    'px-2.5 py-1 rounded text-[11px] font-medium transition-all',
                    colorMode === m ? 'bg-teal/10 text-teal' : 'text-t3 hover:text-t2',
                  )}>
                  {label}
                </button>
              ))}
            </div>

            {activeExpId && st?.status === 'created' && datasetUsable && (
              <button className="btn btn-teal text-[12px] py-1 px-3" onClick={() => initialize.mutate()} disabled={initialize.isPending}>
                {initialize.isPending ? <><RefreshCw className="w-3.5 h-3.5 animate-spin" />Initializing…</> : <><FlaskConical className="w-3.5 h-3.5" />Initialize Experiment</>}
              </button>
            )}
            {activeExpId && canRun && (
              <button className="btn btn-teal text-[12px] py-1 px-3" onClick={() => runIter.mutate()} disabled={runIter.isPending}>
                {runIter.isPending ? <><RefreshCw className="w-3.5 h-3.5 animate-spin" />Running…</> : <><Play className="w-3.5 h-3.5" />Run Iteration</>}
              </button>
            )}
            {runIter.isError && <span className="text-err text-[10px] max-w-40 truncate" title={(runIter.error as any)?.response?.data?.detail}>{(runIter.error as any)?.response?.data?.detail ?? 'Iteration failed'}</span>}
            {runIter.data && <span className="text-ok text-[10px]">Iteration {runIter.data.iteration} complete</span>}
          </div>

          {/* Scatter visualization */}
          <div className="flex-1 min-h-0 px-3 pt-2 pb-0 relative">
            {!activeExpId ? (
              /* No experiment selected → show experiment picker */
              <div className="h-full flex flex-col items-center justify-center gap-4 text-center">
                {!ds || !datasetUsable ? (
                  <>
                    <div className="w-12 h-12 rounded-lg bg-panel border border-line flex items-center justify-center">
                      {datasetsLoading ? <RefreshCw className="w-5 h-5 text-teal animate-spin" /> : <Database className="w-5 h-5 text-t3" />}
                    </div>
                    <div>
                      <div className="text-t2 text-[13px] font-medium">{datasetsLoading ? 'Checking dataset…' : datasetsError ? 'Could not load dataset status' : ds ? 'Dataset metadata is ready, but its feature matrix is empty' : 'No dataset loaded'}</div>
                      <div className="text-t3 text-[12px] mt-0.5">{datasetsError ? (datasetError as Error)?.message : ds?.error_message || 'Reload Matbench Expt Gap or upload a valid composition and target CSV to continue.'}</div>
                    </div>
                    <button className="btn btn-ghost text-[12px]" onClick={() => loadDs.mutate()} disabled={loadDs.isPending || datasetsLoading}>
                      {loadDs.isPending ? 'Preparing…' : ds ? 'Repair Dataset' : 'Load Dataset'}
                    </button>
                    {loadDs.isError && <div className="text-err text-[11px] max-w-md">{(loadDs.error as any)?.response?.data?.detail ?? 'Dataset loading failed. Check the upload format or backend logs.'}</div>}
                  </>
                ) : experimentsLoading ? (
                  <div className="text-t3 text-[12px] flex items-center gap-2"><RefreshCw className="w-4 h-4 animate-spin" />Loading experiments…</div>
                ) : experimentsError ? (
                  <div className="text-center"><div className="text-err text-[12px]">Could not load experiments from the API.</div><button className="btn btn-ghost mt-2 text-[11px]" onClick={() => qc.invalidateQueries({ queryKey: ['experiments-list'] })}>Retry</button></div>
                ) : expList && expList.length > 0 ? (
                  <>
                    <div className="w-12 h-12 rounded-lg bg-panel border border-line flex items-center justify-center">
                      <TrendingUp className="w-5 h-5 text-t3" />
                    </div>
                    <div>
                      <div className="text-t2 text-[13px] font-medium">Select an experiment</div>
                      <div className="text-t3 text-[12px] mt-0.5">Click any recent experiment to visualize its search space</div>
                    </div>
                    <div className="space-y-1.5 w-full max-w-xs">
                      {expList.slice(0, 5).map(e => (
                        <button key={e.id} onClick={() => selectExperiment(e.id)}
                          className="w-full text-left px-3 py-2 rounded card hover:border-teal/30 transition-all flex items-center justify-between gap-3">
                          <div>
                            <div className="text-[12px] font-medium text-t1">{e.name || e.strategy.toUpperCase()}</div>
                            <div className="text-[10px] text-t3 t-mono">{e.id.slice(0,8)}… · {e.status}</div>
                          </div>
                          <ChevronRight className="w-3.5 h-3.5 text-t4" />
                        </button>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-t2 text-[13px] font-medium">No experiments yet</div>
                    <Link to="/setup" className="btn btn-teal text-[12px]"><FlaskConical className="w-3.5 h-3.5" />Create Experiment</Link>
                  </>
                )}
              </div>
            ) : embLoading ? (
              <div className="h-full flex items-center justify-center text-t3 text-[12px] gap-2">
                <RefreshCw className="w-4 h-4 animate-spin" />Computing PCA embedding…
              </div>
            ) : embError ? (
              <div className="h-full flex items-center justify-center text-t3 text-[12px]">
                <div className="text-center">
                  <div className="text-t2 font-medium mb-1">No prediction data yet</div>
                  <div className="text-t3 text-[11px]">Run an iteration to populate the search space visualization.</div>
                  {canRun && (
                    <button className="btn btn-teal text-[12px] mt-3" onClick={() => runIter.mutate()} disabled={runIter.isPending}>
                      <Play className="w-3.5 h-3.5" />Run First Iteration
                    </button>
                  )}
                </div>
              </div>
            ) : embedding ? (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 8, right: 12, bottom: 24, left: 0 }}>
                  <CartesianGrid strokeDasharray="2 6" stroke="rgba(39,51,67,0.5)" />
                  <XAxis dataKey="x" name={embedding.axis_labels[0]} tick={{ fill: '#5C6E82', fontSize: 10 }}
                    label={{ value: embedding.axis_labels[0], position: 'insideBottom', offset: -12, fill: '#5C6E82', fontSize: 10 }} />
                  <YAxis dataKey="y" name={embedding.axis_labels[1]} tick={{ fill: '#5C6E82', fontSize: 10 }}
                    label={{ value: embedding.axis_labels[1], angle: -90, position: 'insideLeft', offset: 10, fill: '#5C6E82', fontSize: 10 }} />
                  <ZAxis range={[36, 36]} />
                  <Tooltip content={<CT />} cursor={false} />
                  <Scatter
                    data={embedding.points}
                    shape={(props: any) => <CustomDot {...props} colorMode={colorMode} />}
                    onClick={(point: any) => setSelectedPointId(point?.candidate_id ?? point?.payload?.candidate_id ?? null)}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            ) : null}

            {/* ID picker overlay when ds loaded but no experiment active */}
            {ds && !activeExpId && expList && expList.length === 0 && (
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10">
                <div className="card flex items-center gap-3 px-4 py-3 shadow-xl" style={{ minWidth: 400 }}>
                  <Info className="w-4 h-4 text-teal flex-shrink-0" />
                  <span className="text-t2 text-[12px]">Paste an experiment ID to visualize</span>
                  <input className="input w-40 text-[12px] py-1" placeholder="exp-id…"
                    value={idInput} onChange={e => setIdInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && idInput.trim() && setActiveExpId(idInput.trim())} />
                  <button className="btn btn-teal text-[12px] py-1" onClick={() => idInput.trim() && setActiveExpId(idInput.trim())}>Load</button>
                </div>
              </div>
            )}
          </div>

          {/* ── Timeline + mini charts ────────────────────────── */}
          <div className="border-t border-line2 flex-shrink-0">

            {/* Activity row */}
            {history && history.length > 0 && (
              <div className="border-b border-line2">
                <div className="flex items-center justify-between px-5 pt-2.5 pb-1.5">
                  <div className="t-section">Recent Activity</div>
                  {activeExpId && (
                    <Link to={`/experiment/${activeExpId}/history`} className="text-t4 hover:text-teal text-[11px] transition-colors">Full history →</Link>
                  )}
                </div>
                <div className="flex items-stretch overflow-x-auto no-scrollbar px-5 pb-3 gap-0">
                  {[...history].reverse().slice(0,6).map((h, i) => (
                    <div key={h.iteration} className="flex items-stretch gap-0">
                      <div className="flex flex-col items-center mr-3">
                        <div className={clsx('w-[6px] h-[6px] rounded-full mt-1', i===0 ? 'bg-teal shadow-[0_0_5px_#59D5CC]' : 'bg-line')} />
                        {i < 5 && <div className="w-px flex-1 bg-line2 my-0.5" />}
                      </div>
                      <div className="pb-2 pr-6 min-w-[150px]">
                        <div className="text-[10px] text-t3 mb-0.5">Iteration {h.iteration} · {h.n_candidates_selected} selected</div>
                        <div className="flex gap-3 text-[11px]">
                          <span className="text-t2">RMSE <span className="t-mono text-vio">{h.test_rmse?.toFixed(3) ?? '—'}</span></span>
                          <span className="text-t2">Best <span className="t-mono text-teal">{h.best_found_value?.toFixed(3) ?? '—'}</span></span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="px-4 pt-3 pb-2">
              <div className="flex justify-between items-center mb-1"><div className="t-section">Active-learning trajectory</div><span className="text-[10px] text-t4">Best observed + held-out RMSE (eV)</span></div>
              {chartData.length ? <ResponsiveContainer width="100%" height={132}>
                <LineChart data={chartData} margin={{ top: 5, right: 12, left: -18, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 5" stroke="rgba(26,44,64,0.8)" vertical={false} />
                  <XAxis dataKey="i" name="Iteration" tick={{ fill: '#8197B2', fontSize: 9 }} />
                  <YAxis tick={{ fill: '#8197B2', fontSize: 9 }} />
                  <Tooltip contentStyle={{ background: '#0A1523', border: '1px solid #1A2C40', borderRadius: 4, fontSize: 11 }} labelFormatter={v => `Iteration ${v}`} />
                  <Legend wrapperStyle={{ fontSize: 10, color: '#8197B2' }} />
                  <Line connectNulls type="monotone" dataKey="best" stroke="#20C7B7" strokeWidth={2} dot={{ r: 2 }} name="Best observed" />
                  <Line connectNulls type="monotone" dataKey="rmse" stroke="#A78BFA" strokeWidth={1.5} dot={{ r: 2 }} name="Held-out RMSE" />
                </LineChart>
              </ResponsiveContainer> : <div className="inset px-3 py-2 text-[11px] text-t3">No iterations recorded. Initialize the experiment and run an iteration to start the trajectory.</div>}
              {chartData.length === 1 && <div className="text-[10px] text-t4">One measured iteration is available; additional runs will establish a trend.</div>}
            </div>
          </div>
        </div>

        {/* ── Right rail: discovery intelligence ──────────────── */}
        <div className="workspace-rail w-[40%] max-w-[620px] min-w-[350px] flex-shrink-0 border-l border-line2 flex flex-col overflow-hidden">
          <div className="px-4 pt-3.5 pb-2 border-b border-line2">
            <div className="t-section">Discovery Intelligence</div>
          </div>
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
            {/* Best found */}
            {st?.best_found_id && (
              <div>
                <div className="text-[11px] text-t3 mb-1.5">Best Discovered</div>
                <div className="t-mono text-[11px] text-t1 font-semibold truncate mb-0.5">{st.best_found_id}</div>
                <div className="flex items-baseline gap-1.5">
                  <span className="t-mono text-amber font-bold text-base">{st.best_found_value?.toFixed(4)}</span>
                  <span className="text-t3 text-[10px]">eV</span>
                </div>
                {st.global_maximum && (
                  <div className="text-[10px] text-t3 mt-0.5">
                    Recovery: {((st.best_found_value! / st.global_maximum) * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            )}

            {/* Latest selection */}
            {embedding?.points && (
              <>
                <hr className="rule2" />
                <div>
                  <div className="text-[11px] text-t3 mb-1.5">Selected This Iteration</div>
                  {embedding.points.filter(p => p.is_selected).slice(0, 6).map(p => (
                    <div key={p.candidate_id} className="flex items-start gap-1.5 mb-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber mt-1.5 flex-shrink-0" />
                      <div className="min-w-0">
                        <div className="t-mono text-[11px] text-t1 truncate">{p.candidate_id}</div>
                        {p.predicted_mean != null && (
                          <div className="text-[10px] text-t3">μ={p.predicted_mean.toFixed(3)} σ={p.predicted_std?.toFixed(3) ?? '—'}</div>
                        )}
                      </div>
                    </div>
                  ))}
                  {embedding.points.filter(p => p.is_selected).length === 0 && (
                    <div className="text-t4 text-[11px]">Run an iteration to see recommendations</div>
                  )}
                </div>
              </>
            )}

            {activeExpId && (
              <>
                <hr className="rule2" />
                <div>
                  <div className="text-[11px] text-t3 mb-1">Predictive uncertainty distribution</div>
                  {uncertaintyBins.length ? <>
                    <ResponsiveContainer width="100%" height={112}>
                      <BarChart data={uncertaintyBins} margin={{ top: 4, right: 2, left: -25, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="2 5" stroke="rgba(26,44,64,0.8)" vertical={false} />
                        <XAxis dataKey="bin" tick={{ fill: '#8197B2', fontSize: 8 }} interval={1} />
                        <YAxis tick={{ fill: '#8197B2', fontSize: 8 }} allowDecimals={false} />
                        <Tooltip contentStyle={{ background: '#0A1523', border: '1px solid #1A2C40', borderRadius: 4, fontSize: 10 }} />
                        <Bar dataKey="count" name="Candidates" fill="#20DCE8" fillOpacity={0.72} />
                      </BarChart>
                    </ResponsiveContainer>
                    <div className="text-[9px] text-t4">Recorded prediction sample · {recommendations?.candidates.length} candidates · σ in target units</div>
                  </> : <div className="inset px-2.5 py-2 text-[10px] text-t3">Not evaluated yet. Run an iteration to calculate pool predictions.</div>}
                </div>
                <hr className="rule2" />
                <div>
                  <div className="text-[11px] text-t3 mb-1">Predicted vs observed</div>
                  {regressionPoints.length ? <>
                    <ResponsiveContainer width="100%" height={135}>
                      <ScatterChart margin={{ top: 5, right: 8, bottom: 12, left: -12 }}>
                        <CartesianGrid strokeDasharray="2 5" stroke="rgba(26,44,64,0.8)" />
                        <XAxis type="number" dataKey="actual" name="Observed" tick={{ fill: '#8197B2', fontSize: 8 }} label={{ value: 'Observed (eV)', position: 'insideBottom', offset: -8, fill: '#8197B2', fontSize: 8 }} />
                        <YAxis type="number" dataKey="predicted" name="Predicted" tick={{ fill: '#8197B2', fontSize: 8 }} />
                        <Tooltip cursor={{ strokeDasharray: '2 2' }} contentStyle={{ background: '#0A1523', border: '1px solid #1A2C40', borderRadius: 4, fontSize: 10 }} formatter={(value: any, name: string, item: any) => [Number(value).toFixed(3), name === 'actual' ? 'Observed (eV)' : `Predicted (eV) · error ${item.payload.error.toFixed(3)}`]} />
                        {(() => { const vals = regressionPoints.flatMap((p: any) => [p.actual, p.predicted]); const lo = Math.min(...vals); const hi = Math.max(...vals); return <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#8197B2" strokeDasharray="4 4" /> })()}
                        <Scatter data={regressionPoints} fill="#20DCE8" fillOpacity={0.75} />
                      </ScatterChart>
                    </ResponsiveContainer>
                    <div className="flex gap-4 text-[10px] text-t3"><span>MAE <b className="t-mono text-vio">{regressionMae?.toFixed(4)} eV</b></span><span>RMSE <b className="t-mono text-vio">{regressionRmse?.toFixed(4)} eV</b></span><span>n={regressionPoints.length}</span></div>
                    <div className="text-[9px] text-t4">Predictions recorded before candidates were acquired; acquisition-selected sample, not an independent test set.</div>
                  </> : <div className="inset px-2.5 py-2 text-[10px] text-t3">Evaluation required. No matched pre-acquisition predictions and observed values are available yet.</div>}
                </div>
                <hr className="rule2" />
                <div>
                  <div className="text-[11px] text-t3 mb-1">Uncertainty calibration</div>
                  {calibrationPlot.length ? <>
                    <ResponsiveContainer width="100%" height={120}>
                      <LineChart data={calibrationPlot} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="2 5" stroke="rgba(26,44,64,0.8)" />
                        <XAxis type="number" dataKey="nominal" domain={[0, 1]} tick={{ fill: '#8197B2', fontSize: 8 }} tickFormatter={v => `${Math.round(v * 100)}%`} />
                        <YAxis type="number" domain={[0, 1]} tick={{ fill: '#8197B2', fontSize: 8 }} tickFormatter={v => `${Math.round(v * 100)}%`} />
                        <Tooltip contentStyle={{ background: '#0A1523', border: '1px solid #1A2C40', borderRadius: 4, fontSize: 10 }} formatter={(v: any, name: string) => [`${(Number(v) * 100).toFixed(1)}%`, name]} />
                        <Legend wrapperStyle={{ fontSize: 9 }} />
                        <Line type="linear" dataKey="ideal" stroke="#526B86" strokeDasharray="4 4" dot={false} name="Ideal" />
                        <Line type="monotone" dataKey="observed" stroke="#A78BFA" strokeWidth={2} dot={{ r: 2 }} name="Observed coverage" />
                      </LineChart>
                    </ResponsiveContainer>
                    <div className="text-[9px] text-t4">n={calibration?.n_aligned} acquired observations · selection-biased estimate</div>
                    <div className="text-[9px] text-t4">{calibration?.caveat}</div>
                  </> : <div className="inset px-2.5 py-2 text-[10px] text-t3">{calibration?.error ?? 'Not evaluated yet. At least five valid pre-acquisition prediction-observation pairs are required.'} {calibration && `Valid pairs: ${calibration.n_aligned}; excluded: ${calibration.n_excluded ?? 0}.`}</div>}
                </div>
                {selectedPointId && (
                  <>
                    <hr className="rule2" />
                    <div className="card p-2.5">
                      <div className="flex justify-between items-center mb-1"><div className="text-[11px] text-t3">Selected material</div><button className="text-t4 hover:text-t1" onClick={() => setSelectedPointId(null)}>×</button></div>
                      <div className="t-mono text-[11px] text-t1 break-all">{pointDetail?.formula || selectedPointId}</div>
                      {pointDetail ? <><div className="text-[10px] text-t3 mt-1">Prediction {pointDetail.predicted_mean.toFixed(4)} eV · σ {pointDetail.predicted_std.toFixed(4)}</div><div className="text-[10px] text-amber mt-1">{pointDetail.selection_reason || 'No selection explanation recorded for this candidate.'}</div></> : <div className="text-[10px] text-t3 mt-1">Prediction details appear after an iteration records this candidate.</div>}
                    </div>
                  </>
                )}
                <div>
                  <div className="text-[11px] text-t3 mb-1.5">Model confidence</div>
                  <div className="space-y-1 text-[10px]">
                    <div className="flex justify-between text-t3"><span>Held-out test RMSE</span><span className="t-mono text-vio">{st?.latest_rmse?.toFixed(4) ?? 'Not evaluated'}</span></div>
                    <div className="flex justify-between text-t3"><span>Held-out MAE</span><span className="t-mono text-vio">{lastHistory?.test_mae?.toFixed(4) ?? 'Not evaluated'}</span></div>
                    <div className="flex justify-between text-t3"><span>90% coverage</span><span className="t-mono text-teal">{calibration && !calibration.error && calibration.coverage_90pct != null ? `${(calibration.coverage_90pct * 100).toFixed(1)}%` : 'Not evaluated'}</span></div>
                    <div className="text-t4 pt-1">{calibration && !calibration.error ? `Pre-acquisition calibration RMSE on ${calibration.n_aligned} acquired observations · selection-biased; small-sample coverage is noisy.` : calibration?.error ?? 'Calibration not evaluated yet.'}</div>
                  </div>
                </div>
                <hr className="rule2" />
                <div>
                  <div className="text-[11px] text-t3 mb-1.5">Next best candidates</div>
                  {(recommendations?.candidates ?? []).filter(c => !c.is_selected).slice(0,4).map((c, i) => (
                    <Link key={c.candidate_id} to={`/experiment/${activeExpId}/candidates/${encodeURIComponent(c.candidate_id)}`} className="flex gap-2 py-1.5 border-b border-line2 last:border-0 group">
                      <span className="t-mono text-[10px] text-teal">{i + 1}</span><span className="min-w-0 flex-1"><span className="block t-mono text-[10px] truncate text-t1 group-hover:text-teal">{c.candidate_id}</span><span className="text-[10px] text-t3">μ {c.predicted_mean.toFixed(3)} · σ {c.predicted_std.toFixed(3)}</span><span className="block text-[9px] text-t4">{(c.breakdown as any)?.selection_reason || `Highest remaining ${activeExperiment?.strategy.toUpperCase() ?? ''} score`}</span></span><span className="t-mono text-[10px] text-amber">{c.acquisition_score.toFixed(2)}</span>
                    </Link>
                  ))}
                  {!recommendations?.candidates?.length && <div className="text-t4 text-[11px]">Run an iteration to generate scores.</div>}
                  {!!recommendations?.candidates?.length && <Link to={`/experiment/${activeExpId}/candidates`} className="text-t4 hover:text-teal text-[10px] mt-2 block">Inspect candidate pool →</Link>}
                </div>
              </>
            )}

            {/* Recent experiments */}
            {expList && expList.length > 0 && (
              <>
                <hr className="rule2" />
                <div>
                  <div className="text-[11px] text-t3 mb-1.5">Recent Experiments</div>
                  <div className="space-y-1">
                    {expList.slice(0, 4).map(e => (
                      <button key={e.id} onClick={() => selectExperiment(e.id)}
                        className={clsx(
                          'w-full text-left p-1.5 rounded transition-all',
                          activeExpId === e.id ? 'bg-teal/10 border-l-2 border-teal pl-[5px]' : 'hover:bg-white/[0.03] border-l-2 border-transparent',
                        )}>
                        <div className="text-[11px] font-medium text-t1 truncate">{e.name || e.strategy.toUpperCase()}</div>
                        <div className="text-[10px] text-t3 t-mono">{e.id.slice(0,8)} · {e.status}</div>
                      </button>
                    ))}
                  </div>
                  <Link to="/history" className="text-t4 hover:text-teal text-[11px] transition-colors mt-2 block">All experiments →</Link>
                </div>
              </>
            )}

            {/* No data state */}
            {!activeExpId && (
              <div className="text-t4 text-[11px] leading-relaxed">
                Select an experiment to see discovery intelligence.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
