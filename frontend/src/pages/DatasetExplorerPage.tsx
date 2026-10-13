import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Database, RefreshCw, FileUp, Search, BarChart2, Table2, Layers } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { api } from '../services/api'
import { Link } from 'react-router-dom'
import StatusBadge from '../components/StatusBadge'
import EmptyState from '../components/EmptyState'
import { SkTable, Sk } from '../components/Skeleton'
import type { Dataset } from '../types'
import clsx from 'clsx'

export default function DatasetExplorerPage() {
  const qc = useQueryClient()
  const [activeId, setActiveId] = useState<string | null>(null)
  const [tab, setTab] = useState<'dist' | 'preview' | 'features'>('dist')
  const [search, setSearch] = useState('')

  const { data: datasets, isLoading } = useQuery({ queryKey: ['datasets'], queryFn: api.datasets.list })
  const { data: experiments } = useQuery({ queryKey: ['experiments-list'], queryFn: () => api.experiments.list(0, 50) })
  const activeExperiment = experiments?.find(e => e.current_iteration > 0) ?? experiments?.[0]
  const ds = datasets?.find(d => d.id === activeId)

  const { data: stats, isLoading: sl } = useQuery({
    queryKey: ['ds-stats', activeId],
    queryFn: () => api.datasets.statistics(activeId!),
    enabled: !!activeId,
  })
  const { data: preview, isLoading: pl } = useQuery({
    queryKey: ['ds-preview', activeId],
    queryFn: () => api.datasets.preview(activeId!, 30),
    enabled: !!activeId && tab === 'preview',
  })

  const load = useMutation({
    mutationFn: () => api.datasets.loadBuiltin('matbench_expt_gap', 'gap expt'),
    onSuccess: (d: any) => { qc.invalidateQueries({ queryKey: ['datasets'] }); d?.id && setActiveId(d.id) },
  })

  const hist = stats?.target_histogram
    ? stats.target_histogram.bin_centers.map((c: number, i: number) => ({ x: +c.toFixed(2), n: stats.target_histogram.counts[i] }))
    : []

  const visible = datasets?.filter(d => !search || d.name.toLowerCase().includes(search.toLowerCase())) ?? []

  return (
    <div className="flex h-full">
      {/* ── Left: dataset list ─────────────────────────────── */}
      <div className="w-[220px] flex-shrink-0 border-r border-line2 flex flex-col">
        <div className="px-3 pt-4 pb-3 border-b border-line2">
          <div className="flex items-center justify-between gap-2 mb-3"><div className="t-title text-[13px]">Materials</div>{activeExperiment && <Link className="text-[10px] text-teal hover:text-t1 whitespace-nowrap" to={`/experiment/${activeExperiment.id}/candidates`}>Model predictions →</Link>}</div>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-t4" />
            <input className="input text-[12px] pl-8 py-1.5" placeholder="Search datasets…"
              value={search} onChange={e => setSearch(e.target.value)} />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
          {isLoading && <div className="space-y-2 px-1">{[0,1].map(i => <Sk key={i} className="h-12 rounded" />)}</div>}
          {!isLoading && visible.length === 0 && (
            <EmptyState icon={<Database className="w-4 h-4" />} title="No datasets" description="Load the built-in dataset." />
          )}
          {visible.map(d => (
            <button key={d.id} onClick={() => { setActiveId(d.id); setTab('dist') }}
              className={clsx(
                'w-full text-left px-3 py-2.5 rounded transition-all',
                d.id === activeId
                  ? 'bg-[rgba(89,213,204,0.08)] border-l-2 border-teal'
                  : 'hover:bg-white/[0.03] border-l-2 border-transparent',
              )}>
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="text-[12px] font-medium text-t1 truncate">{d.name}</span>
                <StatusBadge status={d.status} />
              </div>
              <div className="text-[11px] text-t3">
                {d.num_samples?.toLocaleString() ?? '—'} · {d.num_features ?? '—'} feat
              </div>
            </button>
          ))}
        </div>

        <div className="px-3 pb-3 pt-2 border-t border-line2 space-y-2">
          <button className="btn btn-panel w-full text-[12px]" onClick={() => load.mutate()} disabled={load.isPending}>
            {load.isPending
              ? <><RefreshCw className="w-3.5 h-3.5 animate-spin" />Loading…</>
              : <><Database className="w-3.5 h-3.5" />Load expt_gap</>}
          </button>
          <UploadZone onSuccess={(id) => { qc.invalidateQueries({ queryKey: ['datasets'] }); setActiveId(id) }} />
        </div>
      </div>

      {/* ── Right: dataset detail ──────────────────────────── */}
      <div className="flex-1 min-w-0 flex flex-col overflow-hidden">
        {!ds ? (
          <EmptyState icon={<BarChart2 className="w-5 h-5" />} title="Select a dataset" description="Choose a dataset to inspect its properties and distribution." />
        ) : (
          <>
            {/* Dataset header */}
            <div className="px-5 pt-4 pb-3 border-b border-line2 flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <div className="t-title text-[13px]">{ds.name}</div>
                  <StatusBadge status={ds.status} />
                </div>
                <div className="text-[11px] text-t3 flex gap-3">
                  <span>Source: <span className="text-t2">{ds.source ?? 'matminer'}</span></span>
                  <span>License: <span className="text-teal">CC BY 4.0</span></span>
                  <span>Target: <span className="text-amber">{ds.target_column}</span></span>
                </div>
              </div>
              {stats && (
                <div className="flex gap-5 text-right">
                  <div><div className="t-mono text-teal font-bold">{ds.num_samples?.toLocaleString()}</div><div className="t-label mt-0.5">Samples</div></div>
                  <div><div className="t-mono text-vio font-bold">{ds.num_features}</div><div className="t-label mt-0.5">Features</div></div>
                </div>
              )}
            </div>

            {/* Stat row */}
            {stats && (
              <div className="flex items-stretch border-b border-line2 divide-x divide-line2">
                {[
                  { l: 'Min',    v: stats.target_min.toFixed(3),  u: 'eV', c: 'text-t2'   },
                  { l: 'Max',    v: stats.target_max.toFixed(3),  u: 'eV', c: 'text-amber' },
                  { l: 'Mean',   v: stats.target_mean.toFixed(3), u: 'eV', c: 'text-teal'  },
                  { l: 'Std',    v: stats.target_std.toFixed(3),  u: 'eV', c: 'text-vio'   },
                  { l: 'Median', v: stats.target_median.toFixed(3), u: 'eV', c: 'text-t2'  },
                ].map(({ l, v, u, c }) => (
                  <div key={l} className="px-5 py-3">
                    <div className="t-label mb-0.5">{l}</div>
                    <div className={clsx('t-mono font-bold text-sm', c)}>{v} <span className="text-t3 font-normal text-[11px]">{u}</span></div>
                  </div>
                ))}
              </div>
            )}

            {/* Tabs */}
            <div className="flex items-center gap-0 border-b border-line2 px-5">
              {[{ k: 'dist', label: 'Distribution' }, { k: 'preview', label: 'Preview' }, { k: 'features', label: 'Features' }].map(t => (
                <button key={t.k} onClick={() => setTab(t.k as any)}
                  className={clsx(
                    'px-4 py-2.5 text-[12px] font-medium transition-colors border-b-2 -mb-px',
                    tab === t.k ? 'border-teal text-teal' : 'border-transparent text-t3 hover:text-t2',
                  )}>
                  {t.label}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div className="flex-1 overflow-auto p-5">
              {tab === 'dist' && (
                <div>
                  <div className="t-label mb-4">Target Distribution — {ds.target_column}</div>
                  {sl ? <Sk className="h-52 rounded" /> : hist.length > 0 ? (
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={hist} margin={{ top: 4, right: 8, left: -14, bottom: 14 }}>
                        <CartesianGrid strokeDasharray="2 5" stroke="rgba(39,51,67,0.6)" vertical={false} />
                        <XAxis dataKey="x" tick={{ fill: '#5C6E82', fontSize: 10 }}
                          label={{ value: 'Band Gap (eV)', position: 'insideBottom', offset: -8, fill: '#5C6E82', fontSize: 10 }} />
                        <YAxis tick={{ fill: '#5C6E82', fontSize: 10 }} />
                        <Tooltip contentStyle={{ background: '#171F2B', border: '1px solid #273343', borderRadius: 4, fontSize: 11 }}
                          formatter={(v: number) => [v.toLocaleString(), 'Candidates']}
                          labelFormatter={(l: number) => `${l} eV`} />
                        <Bar dataKey="n" radius={[2, 2, 0, 0]}>
                          {hist.map((_: any, i: number) => <Cell key={i} fill={`hsl(${186+i*3},60%,${36+i*0.6}%)`} opacity={0.85} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : <div className="text-t3 text-[12px]">No data</div>}
                </div>
              )}

              {tab === 'preview' && (
                <div>
                  {pl ? <SkTable rows={8} /> : preview ? (
                    <div className="overflow-x-auto">
                      <table className="tbl min-w-full">
                        <thead><tr>{preview.columns.slice(0,7).map((c: string) => <th key={c}>{c}</th>)}</tr></thead>
                        <tbody>
                          {preview.rows.slice(0,20).map((row: any, i: number) => (
                            <tr key={i}>{preview.columns.slice(0,7).map((c: string) => (
                              <td key={c} className="t-mono">
                                {typeof row[c] === 'number' ? row[c].toFixed(4) : String(row[c] ?? '—').slice(0,22)}
                              </td>
                            ))}</tr>
                          ))}
                        </tbody>
                      </table>
                      <div className="text-t3 text-[11px] mt-2">Showing 20 of {preview.total_rows.toLocaleString()} rows</div>
                    </div>
                  ) : null}
                </div>
              )}

              {tab === 'features' && stats && (
                <div className="overflow-x-auto">
                  <table className="tbl">
                    <thead><tr><th>Feature</th><th className="text-right">Mean</th><th className="text-right">Std</th></tr></thead>
                    <tbody>
                      {stats.feature_names.slice(0,50).map((n: string, i: number) => (
                        <tr key={n}>
                          <td className="t-mono text-[11px]">{n}</td>
                          <td className="t-mono text-right text-[11px]">{(stats as any).feature_means?.[i]?.toFixed(4) ?? '—'}</td>
                          <td className="t-mono text-right text-[11px]">{(stats as any).feature_stds?.[i]?.toFixed(4) ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function UploadZone({ onSuccess }: { onSuccess: (id: string) => void }) {
  const [up, setUp] = useState(false)
  const [err, setErr] = useState('')
  async function handle(f: File) {
    if (!f.name.endsWith('.csv')) { setErr('CSV only'); return }
    setUp(true); setErr('')
    try {
      const form = new FormData(); form.append('file', f); form.append('target_column', 'gap expt')
      const r = await api.datasets.upload(form)
      if (r?.id) onSuccess(r.id)
    } catch (e: any) { setErr(e?.response?.data?.detail ?? 'Upload failed') }
    finally { setUp(false) }
  }
  return (
    <div className="border border-dashed border-line rounded p-3 text-center cursor-pointer hover:border-teal/40 transition-colors"
      onDragOver={e => e.preventDefault()}
      onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; f && handle(f) }}
      onClick={() => { const i = document.createElement('input'); i.type='file'; i.accept='.csv'; i.onchange=(e:any)=>{ const f=e.target.files?.[0]; f&&handle(f) }; i.click() }}>
      {up ? <span className="text-[11px] text-teal">Featurizing…</span>
          : <><FileUp className="w-4 h-4 text-t4 mx-auto mb-1" /><div className="text-t3 text-[11px]">Drop CSV or click</div></>}
      {err && <div className="text-err text-[10px] mt-1">{err}</div>}
    </div>
  )
}
