import { useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Clock, Award, Download, ExternalLink, Search, Filter } from 'lucide-react'
import { api } from '../services/api'
import type { Experiment } from '../types'
import Sparkline from '../components/Sparkline'
import { SkTable } from '../components/Skeleton'
import EmptyState from '../components/EmptyState'
import StatusBadge from '../components/StatusBadge'
import clsx from 'clsx'

const SL: Record<string, string> = {
  random: 'Random', uncertainty: 'Uncertainty', ei: 'EI', ucb: 'UCB', puda: 'PUDA',
}
const SC: Record<string, string> = {
  random: '#3A4F64', uncertainty: '#59D5CC', ei: '#A99BFF', ucb: '#EDAB60', puda: '#41D98C',
}

function ExperimentRow({ exp, datasetName }: { exp: Experiment; datasetName?: string }) {
  const { data: history } = useQuery({
    queryKey: ['exp-history-mini', exp.id],
    queryFn: () => api.experiments.history(exp.id),
    enabled: exp.current_iteration > 0,
    staleTime: 60_000,
  })

  const rmses = history?.map(h => h.test_rmse ?? 0) ?? []
  const bests = history?.map(h => h.best_found_value ?? 0) ?? []

  async function downloadReport(fmt: 'json' | 'csv') {
    const blob = await api.experiments.report(exp.id, fmt)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url
    a.download = `experiment_${exp.id.slice(0,8)}_report.${fmt}`; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <tr className="group hover:bg-white/[0.015] cursor-pointer">
      <td>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-sm flex-shrink-0" style={{ background: SC[exp.strategy] ?? '#888' }} />
          <div>
            <div className="text-[12px] font-medium text-t1">{exp.name || `${SL[exp.strategy] ?? exp.strategy} experiment`}</div>
            <div className="t-mono text-[10px] text-t3">{exp.id.slice(0,8)}…</div>
          </div>
        </div>
      </td>
      <td>
        <StatusBadge status={exp.status} />
      </td>
      <td className="text-[11px] text-t2">{datasetName ?? <span className="text-err">Unavailable</span>}</td>
      <td>
        <span className="chip text-[10px]" style={{ background: `${SC[exp.strategy] ?? '#888'}18`, color: SC[exp.strategy] ?? '#888', border: `1px solid ${SC[exp.strategy] ?? '#888'}28` }}>
          {SL[exp.strategy] ?? exp.strategy}
        </span>
      </td>
      <td className="t-mono text-[11px]">{exp.model_type.toUpperCase()}</td>
      <td className="t-mono text-right">{exp.current_iteration}</td>
      <td>
        <div className="flex items-center gap-2">
          <div className="w-16 h-5 flex-shrink-0">
            <div className="flex items-center h-1.5 bg-panel3 rounded-sm overflow-hidden">
              <div style={{
                width: `${Math.min(100, (exp.current_budget_used / exp.total_budget) * 100)}%`,
                background: '#59D5CC', opacity: 0.7,
              }} className="h-full" />
            </div>
          </div>
          <span className="t-mono text-[10px] text-t3">{exp.current_budget_used}/{exp.total_budget}</span>
        </div>
      </td>
      <td className="t-mono text-right font-semibold text-amber">{exp.best_found_value?.toFixed(4) ?? '—'}</td>
      <td>
        {rmses.length > 1 ? <Sparkline data={rmses} color="#A99BFF" inverted width={60} height={22} /> : <span className="text-t4 text-[11px]">—</span>}
      </td>
      <td>
        {bests.length > 1 ? <Sparkline data={bests} color="#59D5CC" width={60} height={22} /> : <span className="text-t4 text-[11px]">—</span>}
      </td>
      <td className="t-mono text-[10px] text-t3 whitespace-nowrap">
        {new Date(exp.updated_at).toLocaleDateString()}
      </td>
      <td>
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <Link to={`/experiment/${exp.id}`} className="btn btn-ghost text-[11px] py-0.5 px-2">
            <ExternalLink className="w-3 h-3" />Open
          </Link>
          {exp.current_iteration > 0 && (
            <div className="relative group/dl">
              <button className="btn btn-ghost text-[11px] py-0.5 px-2">
                <Download className="w-3 h-3" />
              </button>
              <div className="absolute right-0 top-full mt-1 bg-panel border border-line rounded shadow-lg z-10 hidden group-hover/dl:block min-w-[100px]">
                <button onClick={() => downloadReport('json')} className="block w-full px-3 py-1.5 text-left text-[11px] hover:bg-white/[0.04] text-t2">JSON report</button>
                <button onClick={() => downloadReport('csv')}  className="block w-full px-3 py-1.5 text-left text-[11px] hover:bg-white/[0.04] text-t2">CSV history</button>
              </div>
            </div>
          )}
        </div>
      </td>
    </tr>
  )
}

export default function ExperimentHistoryPage() {
  const { id } = useParams<{ id: string }>()  // might be in /experiment/:id/history
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [page, setPage] = useState(0)
  const PAGE = 20

  // If we have a specific experiment ID, show its iteration history
  if (id) return <ExperimentIterationHistory id={id} />

  // Otherwise show the global experiment list
  const { data: allExps, isLoading } = useQuery({
    queryKey: ['experiments-list-full', page],
    queryFn: () => api.experiments.list(page * PAGE, PAGE),
  })
  const { data: datasets } = useQuery({ queryKey: ['datasets'], queryFn: api.datasets.list })

  const filtered = allExps?.filter(e => {
    if (search && !e.name?.toLowerCase().includes(search.toLowerCase()) && !e.id.includes(search)) return false
    if (filterStatus !== 'all' && e.status !== filterStatus) return false
    return true
  }) ?? []

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-line2 bg-ink1">
        <div>
          <div className="t-title text-[13px]">Experiment History</div>
          <div className="text-t3 text-[11px] mt-0.5">{allExps?.length ?? 0} experiments</div>
        </div>
        <Link to="/setup" className="btn btn-teal text-[12px]">+ New Experiment</Link>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 px-5 py-2.5 border-b border-line2 flex-wrap">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-t4" />
          <input className="input text-[12px] pl-8 py-1.5 w-52" placeholder="Search experiments…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="flex gap-0.5 bg-ink1 border border-line2 rounded p-0.5">
          {['all', 'running', 'completed', 'initialized', 'created'].map(s => (
            <button key={s} onClick={() => setFilterStatus(s)}
              className={clsx('px-2.5 py-1 rounded text-[11px] font-medium capitalize transition-all',
                filterStatus === s ? 'bg-teal/10 text-teal' : 'text-t3 hover:text-t2')}>
              {s}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        {isLoading ? <SkTable rows={8} /> : filtered.length === 0 ? (
          <EmptyState
            icon={<Clock className="w-5 h-5" />}
            title="No experiments yet"
            description="Create your first active learning experiment."
            action={{ label: 'New Experiment', to: '/setup' }}
          />
        ) : (
          <table className="tbl w-full">
            <thead>
              <tr>
                <th>Name / ID</th>
                <th>Status</th>
                <th>Dataset</th>
                <th>Strategy</th>
                <th>Model</th>
                <th className="text-right">Iterations</th>
                <th>Budget</th>
                <th className="text-right">Best Found</th>
                <th>RMSE ↓</th>
                <th>Best ↑</th>
                <th>Updated</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {filtered.map(e => <ExperimentRow key={e.id} exp={e} datasetName={datasets?.find(d => d.id === e.dataset_id)?.name} />)}
            </tbody>
          </table>
        )}
      </div>

      {allExps && allExps.length === PAGE && (
        <div className="flex items-center justify-between px-5 py-2.5 border-t border-line2 text-[12px]">
          <button className="btn btn-ghost text-[11px] py-1" onClick={() => setPage(p => Math.max(0, p-1))} disabled={page===0}>← Prev</button>
          <span className="text-t3">Page {page + 1}</span>
          <button className="btn btn-ghost text-[11px] py-1" onClick={() => setPage(p => p+1)}>Next →</button>
        </div>
      )}
    </div>
  )
}

// ── Iteration history for a specific experiment ───────────────────────────────
function ExperimentIterationHistory({ id }: { id: string }) {
  const { data: history, isLoading } = useQuery({
    queryKey: ['experiment-history', id],
    queryFn: () => api.experiments.history(id),
    enabled: !!id, refetchInterval: 10_000,
  })

  const rmses  = history?.map(h => h.test_rmse  ?? 0) ?? []
  const bests  = history?.map(h => h.best_found_value ?? 0) ?? []
  const regrets = history?.map(h => h.regret ?? 0) ?? []
  const maxBest = history ? Math.max(...history.map(h => h.best_found_value ?? 0)) : 0

  async function downloadReport(fmt: 'json' | 'csv') {
    const blob = await api.experiments.report(id, fmt)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url
    a.download = `experiment_${id.slice(0,8)}_report.${fmt}`; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-line2 bg-ink1">
        <div>
          <div className="t-title text-[13px]">Iteration History</div>
          <div className="text-t3 text-[11px] mt-0.5 t-mono">{id.slice(0,8)}… · {history?.length ?? 0} iterations</div>
        </div>
        <div className="flex gap-2">
          {history && history.length > 0 && (
            <>
              <button className="btn btn-ghost text-[12px]" onClick={() => downloadReport('csv')}>
                <Download className="w-3.5 h-3.5" />CSV
              </button>
              <button className="btn btn-ghost text-[12px]" onClick={() => downloadReport('json')}>
                <Download className="w-3.5 h-3.5" />JSON Report
              </button>
            </>
          )}
        </div>
      </div>

      {/* Sparkline summary */}
      {history && history.length > 1 && (
        <div className="flex items-stretch border-b border-line2 divide-x divide-line2">
          {[
            { l: 'RMSE ↓', data: rmses,   color: '#A99BFF', inv: true  },
            { l: 'Best ↑', data: bests,   color: '#59D5CC', inv: false },
            { l: 'Regret ↓', data: regrets, color: '#EDAB60', inv: true },
          ].map(({ l, data, color, inv }) => (
            <div key={l} className="px-5 py-3 flex items-center gap-4">
              <div>
                <div className="t-label mb-0.5">{l}</div>
                <div className="t-mono font-bold text-sm" style={{ color }}>{data[data.length-1]?.toFixed(4) ?? '—'}</div>
              </div>
              <Sparkline data={data} color={color} inverted={inv} width={70} height={28} />
            </div>
          ))}
        </div>
      )}

      <div className="flex-1 overflow-auto">
        {isLoading ? <SkTable rows={8} /> : !history?.length ? (
          <EmptyState title="No iterations yet" description="Run the active learning console to see per-iteration metrics here." />
        ) : (
          <table className="tbl w-full">
            <thead>
              <tr>
                <th>Iter</th>
                <th className="text-right">Budget</th>
                <th className="text-right">Remaining</th>
                <th className="text-right">MAE</th>
                <th className="text-right">RMSE</th>
                <th className="text-right">R²</th>
                <th className="text-right">Best Found</th>
                <th className="text-right">Regret</th>
                <th className="text-right">Diversity</th>
                <th className="text-right">Selected</th>
                <th className="text-right">Time</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, i) => {
                const isLast = i === history.length - 1
                const isBest = h.best_found_value === maxBest && maxBest > 0
                return (
                  <tr key={h.iteration} className={isLast ? 'bg-[rgba(89,213,204,0.02)]' : ''}>
                    <td>
                      <div className="flex items-center gap-1.5">
                        {isBest && <Award className="w-3 h-3 text-amber flex-shrink-0" />}
                        <span className="t-mono font-bold text-t1">{h.iteration}</span>
                        {isLast && <span className="chip chip-teal text-[10px]">latest</span>}
                      </div>
                    </td>
                    <td className="t-mono text-right">{h.budget_used}</td>
                    <td className="t-mono text-right text-t3">{h.budget_remaining}</td>
                    <td className="t-mono text-right text-vio">{h.test_mae?.toFixed(4) ?? '—'}</td>
                    <td className="t-mono text-right text-err">{h.test_rmse?.toFixed(4) ?? '—'}</td>
                    <td className="t-mono text-right text-ok">{h.test_r2?.toFixed(4) ?? '—'}</td>
                    <td className={clsx('t-mono text-right font-semibold', isBest ? 'text-amber' : 'text-t1')}>
                      {h.best_found_value?.toFixed(4) ?? '—'}
                    </td>
                    <td className="t-mono text-right text-teal">{h.regret?.toFixed(4) ?? '—'}</td>
                    <td className="t-mono text-right text-t3">{h.diversity_score?.toFixed(4) ?? '—'}</td>
                    <td className="t-mono text-right">{h.n_candidates_selected}</td>
                    <td className="t-mono text-right text-t3">{h.iteration_time_seconds?.toFixed(1) ?? '—'}s</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
