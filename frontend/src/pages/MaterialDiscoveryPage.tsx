import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Trophy, Download, Medal } from 'lucide-react'
import { api } from '../services/api'
import type { ExperimentStatusResponse } from '../types'
import { SkTable } from '../components/Skeleton'
import EmptyState from '../components/EmptyState'
import clsx from 'clsx'

const SC: Record<string, string> = { random:'#3A4F64', uncertainty:'#59D5CC', ei:'#A99BFF', ucb:'#EDAB60', puda:'#41D98C' }
const SL: Record<string, string> = { random:'Random', uncertainty:'Unc', ei:'EI', ucb:'UCB', puda:'PUDA' }

function Rank({ rank }: { rank: number }) {
  if (rank === 1) return <Trophy className="w-3.5 h-3.5 text-amber" />
  if (rank === 2) return <Medal  className="w-3.5 h-3.5 text-slate-300" />
  if (rank === 3) return <Medal  className="w-3.5 h-3.5 text-amber/50" />
  return <span className="t-mono text-[11px] text-t3">#{rank}</span>
}

export default function MaterialDiscoveryPage() {
  const { id } = useParams<{ id: string }>()

  const { data: mats, isLoading } = useQuery({
    queryKey: ['discovery', id],
    queryFn: () => api.experiments.discovery(id!, 100),
    enabled: !!id, refetchInterval: 10_000,
  })

  const { data: st } = useQuery<ExperimentStatusResponse>({
    queryKey: ['experiment-status', id],
    queryFn: () => api.experiments.status(id!),
    enabled: !!id,
  })

  async function handleExport() {
    if (!id) return
    const data = await api.experiments.export(id)
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = `exp_${id.slice(0,8)}_results.json`; a.click()
  }

  const top3 = (mats ?? []).slice(0,3)
  const rec  = st?.global_maximum && st.best_found_value
    ? ((st.best_found_value / st.global_maximum) * 100).toFixed(1) : null

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-line2 bg-ink1">
        <div>
          <div className="t-title text-[13px]">Material Discovery</div>
          <div className="text-t3 text-[11px] mt-0.5">Highest-performing materials discovered through active learning</div>
        </div>
        <button className="btn btn-ghost text-[12px]" onClick={handleExport}>
          <Download className="w-3.5 h-3.5" />Export
        </button>
      </div>

      {/* Summary strip */}
      {st && (
        <div className="flex items-stretch border-b border-line2">
          {[
            { l: 'Best Discovered', v: st.best_found_value?.toFixed(4) ?? '—', c: '#EDAB60', hint: 'eV' },
            { l: 'Global Maximum',  v: st.global_maximum?.toFixed(4) ?? '—',   c: '#9AA9BA', hint: 'eV' },
            { l: 'Recovery Rate',   v: rec ? `${rec}%` : '—',                  c: '#41D98C', hint: 'best/max' },
            { l: 'Discovered',      v: String(mats?.length ?? 0),              c: '#59D5CC', hint: 'candidates' },
          ].map(({ l, v, c, hint }) => (
            <div key={l} className="stat-item py-3 first:pl-5">
              <span className="t-label">{l}</span>
              <span className="stat-val" style={{ color: c }}>{v}{' '}<span className="text-t3 font-normal text-[11px]">{hint}</span></span>
            </div>
          ))}
        </div>
      )}

      <div className="flex-1 overflow-auto">
        {/* Podium */}
        {top3.length > 0 && (
          <div className="grid grid-cols-3 gap-4 px-5 py-4 border-b border-line2">
            {top3.map((m: any) => (
              <div key={m.rank} className={clsx('card p-4', m.rank === 1 && 'border-amber/30 bg-[rgba(237,171,96,0.04)]')}>
                <div className="flex items-center justify-between mb-2.5">
                  <Rank rank={m.rank} />
                  <span className="chip text-[10px]" style={{ background:`${SC[m.strategy]??'#888'}18`, color:SC[m.strategy]??'#888', border:`1px solid ${SC[m.strategy]??'#888'}28` }}>
                    {SL[m.strategy] ?? m.strategy}
                  </span>
                </div>
                <div className="t-mono text-[12px] text-t1 font-semibold mb-1 truncate">{m.candidate_id}</div>
                <div className="flex items-baseline gap-1.5">
                  <span className={clsx('t-mono font-bold text-[1.4rem]', m.rank === 1 ? 'text-amber' : 'text-t1')}>{m.true_value.toFixed(4)}</span>
                  <span className="text-t3 text-[11px]">eV</span>
                </div>
                <div className="text-t3 text-[10px] mt-2">Iteration {m.iteration_discovered}</div>
              </div>
            ))}
          </div>
        )}

        {/* Full table */}
        {isLoading ? <SkTable rows={8} /> : !(mats?.length) ? (
          <EmptyState icon={<Trophy className="w-5 h-5" />} title="No materials discovered yet"
            description="Run active learning iterations to discover high-performing materials." />
        ) : (
          <table className="tbl w-full">
            <thead>
              <tr>
                <th>Rank</th><th>Material ID</th>
                <th className="text-right">Band Gap ↑</th>
                <th className="text-right">Predicted</th>
                <th className="text-center">Strategy</th>
                <th className="text-center">Iteration</th>
                <th className="text-right">Budget</th>
              </tr>
            </thead>
            <tbody>
              {mats.map((m: any) => (
                <tr key={m.rank} className={m.rank <= 3 ? 'bg-[rgba(237,171,96,0.03)]' : ''}>
                  <td><Rank rank={m.rank} /></td>
                  <td className="t-mono text-[11px] text-t1 font-medium">{m.candidate_id}</td>
                  <td className={clsx('t-mono text-right font-bold', m.rank===1 ? 'text-amber' : m.rank<=3 ? 'text-t1' : 'text-t2')}>
                    {m.true_value.toFixed(4)}
                  </td>
                  <td className="t-mono text-right text-t3">{m.predicted_mean?.toFixed(4) ?? '—'}</td>
                  <td className="text-center">
                    <span className="chip text-[10px]" style={{ background:`${SC[m.strategy]??'#888'}18`, color:SC[m.strategy]??'#888', border:`1px solid ${SC[m.strategy]??'#888'}28` }}>
                      {SL[m.strategy] ?? m.strategy}
                    </span>
                  </td>
                  <td className="t-mono text-center">{m.iteration_discovered}</td>
                  <td className="t-mono text-right text-t3">{m.budget_at_discovery || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="border-t border-line2 px-5 py-3 text-[11px] text-t4">
        <span className="text-amber font-medium">Simulated evaluation:</span> True values are experimental measurements from the held-out dataset, revealed only after each candidate was selected.
      </div>
    </div>
  )
}
