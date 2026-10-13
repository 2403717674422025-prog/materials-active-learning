import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Search, ArrowUpDown, ChevronRight } from 'lucide-react'
import { api } from '../services/api'
import UncertaintyBadge from '../components/UncertaintyBadge'
import { SkTable } from '../components/Skeleton'
import EmptyState from '../components/EmptyState'
import clsx from 'clsx'

type S = 'acquisition_score' | 'predicted_mean' | 'predicted_std'
type F = 'all' | 'selected' | 'unselected'

export default function CandidateExplorerPage() {
  const { id } = useParams<{ id: string }>()
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState<S>('acquisition_score')
  const [dir, setDir] = useState<'desc'|'asc'>('desc')
  const [filter, setFilter] = useState<F>('all')
  const [page, setPage] = useState(1)

  const { data, isLoading } = useQuery({
    queryKey: ['experiment-candidates', id, page],
    queryFn: () => api.experiments.candidates(id!, undefined, page, 100),
    enabled: !!id,
  })

  const sorted = (data?.candidates ?? [])
    .filter(c => (!search || c.candidate_id.toLowerCase().includes(search.toLowerCase()))
      && (filter === 'all' || (filter === 'selected') === c.is_selected))
    .sort((a, b) => (b[sort] - a[sort]) * (dir === 'desc' ? 1 : -1))

  function th(key: S, label: string) {
    return (
      <button className="flex items-center gap-1 hover:text-teal transition-colors" onClick={() => {
        if (sort === key) setDir(d => d === 'desc' ? 'asc' : 'desc')
        else { setSort(key); setDir('desc') }
      }}>
        {label}
        <ArrowUpDown className={clsx('w-3 h-3', sort === key ? 'text-teal' : 'text-t4')} />
      </button>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Controls */}
      <div className="flex items-center gap-3 px-5 py-3 border-b border-line2 bg-ink1 flex-wrap">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-t4" />
          <input className="input text-[12px] pl-8 py-1.5 w-52" placeholder="Search material ID…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="flex gap-0.5 bg-ink1 border border-line2 rounded p-0.5">
          {(['all','selected','unselected'] as F[]).map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={clsx('px-3 py-1 rounded text-[11px] font-medium capitalize transition-all',
                filter === f ? 'bg-teal/10 text-teal' : 'text-t3 hover:text-t2')}>
              {f}
            </button>
          ))}
        </div>
        <span className="text-t3 text-[12px] ml-auto">{sorted.length.toLocaleString()} candidates</span>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        {isLoading ? <SkTable rows={12} /> : sorted.length === 0 ? (
          <EmptyState icon={<Search className="w-5 h-5" />} title="No candidates" description="Adjust your search or run an iteration first." />
        ) : (
          <table className="tbl w-full">
            <thead>
              <tr>
                <th>Material ID</th>
                <th className="text-right">{th('predicted_mean', 'Predicted (eV)')}</th>
                <th className="text-right">95% interval (GPR)</th>
                <th className="text-center">{th('predicted_std', 'Uncertainty')}</th>
                <th className="text-right">{th('acquisition_score', 'Score')}</th>
                <th className="text-center">Status</th>
                <th className="text-center">Rank</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0,200).map(c => (
                <tr key={c.candidate_id}
                  className={clsx(c.is_selected && 'bg-[rgba(237,171,96,0.04)]')}>
                  <td className="t-mono text-[11px] text-t1 font-medium">{c.candidate_id}</td>
                  <td className="t-mono text-right font-semibold text-t1">{c.predicted_mean.toFixed(4)}</td>
                  <td className="t-mono text-right text-t3 text-[11px]">{c.lower_cb == null || c.upper_cb == null ? 'Not calibrated' : `[${c.lower_cb.toFixed(2)},${c.upper_cb.toFixed(2)}]`}</td>
                  <td className="text-center"><UncertaintyBadge label={c.uncertainty_label} value={c.predicted_std} showValue /></td>
                  <td className="t-mono text-right font-semibold" style={{ color: '#59D5CC' }}>{c.acquisition_score.toFixed(4)}</td>
                  <td className="text-center">
                    {c.is_selected ? <span className="chip chip-amber">Selected</span> : <span className="text-t4 text-[11px]">—</span>}
                  </td>
                  <td className="t-mono text-center text-t3 text-[11px]">{c.selection_rank ?? '—'}</td>
                  <td>
                    <Link to={`/experiment/${id}/candidates/${encodeURIComponent(c.candidate_id)}`}
                      className="text-t4 hover:text-teal transition-colors">
                      <ChevronRight className="w-3.5 h-3.5" />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {data && data.total_pages > 1 && (
        <div className="flex items-center justify-between px-5 py-2.5 border-t border-line2 text-[12px]">
          <span className="text-t3">Page {page}/{data.total_pages}</span>
          <div className="flex gap-2">
            <button className="btn btn-ghost text-[11px] py-1" onClick={() => setPage(p => Math.max(1,p-1))} disabled={page===1}>Prev</button>
            <button className="btn btn-ghost text-[11px] py-1" onClick={() => setPage(p => Math.min(data.total_pages,p+1))} disabled={page===data.total_pages}>Next</button>
          </div>
        </div>
      )}
    </div>
  )
}
