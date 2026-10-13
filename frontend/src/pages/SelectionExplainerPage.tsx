import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft } from 'lucide-react'
import { api } from '../services/api'
import AcquisitionBreakdown from '../components/AcquisitionBreakdown'
import UncertaintyBadge from '../components/UncertaintyBadge'
import { SkCard } from '../components/Skeleton'

export default function SelectionExplainerPage() {
  const { id, candidateId } = useParams<{ id: string; candidateId: string }>()
  const { data: d, isLoading } = useQuery({
    queryKey: ['candidate-detail', id, candidateId],
    queryFn: () => api.experiments.candidateDetail(id!, decodeURIComponent(candidateId!)),
    enabled: !!(id && candidateId),
  })

  if (isLoading) return <div className="p-6 grid grid-cols-2 gap-4">{[0,1,2,3].map(i=><SkCard key={i}/>)}</div>
  if (!d) return <div className="p-6 text-t3 text-[12px]">Candidate not found.</div>

  const bd = d.full_breakdown as Record<string, number>
  const isPuda = d.strategy?.toLowerCase().includes('puda')
  const σ = d.predicted_std ?? 0
  const ul: 'Low'|'Medium'|'High' = σ < 0.3 ? 'Low' : σ < 0.8 ? 'Medium' : 'High'

  return (
    <div className="flex h-full">
      {/* Main */}
      <div className="flex-1 min-w-0 overflow-auto px-6 py-6">
        <Link to={`/experiment/${id}/candidates`}
          className="inline-flex items-center gap-1.5 text-t3 hover:text-teal text-[12px] mb-5 transition-colors">
          <ArrowLeft className="w-3.5 h-3.5" />Back to Candidates
        </Link>

        <div className="flex items-start justify-between gap-4 mb-6">
          <div>
            <div className="t-label text-teal mb-1">Selection Explanation</div>
            <div className="t-mono text-[14px] font-semibold text-t1 break-all">{d.candidate_id}</div>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {d.is_selected
              ? <><span className="chip chip-amber">Selected</span>{d.selection_rank && <span className="chip chip-idle">Rank #{d.selection_rank}</span>}</>
              : <span className="chip chip-idle">Not Selected</span>
            }
          </div>
        </div>

        {/* Prediction */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="card p-4">
            <div className="t-label mb-2">Predicted Band Gap</div>
            <div className="flex items-baseline gap-2">
              <span className="t-mono text-[1.6rem] font-bold text-t1">{d.predicted_mean.toFixed(4)}</span>
              <span className="text-t3 text-[12px]">eV</span>
            </div>
            <div className="text-t3 text-[11px] mt-1.5">
              95% interval (GPR): {d.lower_cb == null || d.upper_cb == null ? 'Not calibrated' : `[${d.lower_cb.toFixed(3)}, ${d.upper_cb.toFixed(3)}] eV`}
            </div>
          </div>
          <div className="card p-4">
            <div className="t-label mb-2">Predictive Uncertainty σ</div>
            <div className="flex items-baseline gap-2">
              <span className="t-mono text-[1.6rem] font-bold text-t1">{d.predicted_std.toFixed(4)}</span>
              <span className="text-t3 text-[12px]">eV</span>
            </div>
            <div className="mt-2"><UncertaintyBadge label={ul} value={d.predicted_std} showValue /></div>
            <div className="text-t3 text-[11px] mt-1.5">True GP posterior σ — not a heuristic</div>
          </div>
        </div>

        {/* Acquisition */}
        <div className="card p-4 mb-4">
          <div className="t-label mb-3">Acquisition Score — {d.strategy}</div>
          {isPuda ? (
            <AcquisitionBreakdown
              alpha={bd.alpha ?? 0.4} beta={bd.beta ?? 0.35} gamma={bd.gamma ?? 0.25}
              predNorm={bd.prediction_normalized ?? 0}
              uncNorm={bd.uncertainty_normalized ?? 0}
              divNorm={bd.diversity_normalized ?? 0}
              finalScore={d.acquisition_score}
              strategy={d.strategy}
            />
          ) : (
            <div className="space-y-2">
              {[
                { l: 'Performance',  v: d.exploitation_contribution, c: '#59D5CC' },
                { l: 'Exploration',  v: d.exploration_contribution,  c: '#A99BFF' },
                { l: 'Diversity',    v: d.diversity_contribution,    c: '#EDAB60' },
              ].filter(r => r.v != null).map(({ l, v, c }) => (
                <div key={l} className="flex items-center gap-3">
                  <span className="text-t3 text-[11px] w-20">{l}</span>
                  <div className="flex-1 h-1 bg-panel3 rounded-sm overflow-hidden">
                    <div style={{ width: `${Math.min(100,(v??0)*100)}%`, background: c, opacity: 0.7 }} className="h-full" />
                  </div>
                  <span className="t-mono text-[11px] text-t2 w-14 text-right">{(v??0).toFixed(4)}</span>
                </div>
              ))}
              <div className="flex justify-between pt-2 border-t border-line2 text-[12px]">
                <span className="text-t2 font-medium">Final Score</span>
                <span className="t-mono text-teal font-bold">{d.acquisition_score.toFixed(4)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Reason */}
        {d.selection_reason && (
          <div className="card p-4">
            <div className="t-label mb-2">Selection Reasoning</div>
            <p className="text-t2 text-[12px] leading-relaxed">{d.selection_reason}</p>
            {d.formula && (
              <div className="inset px-3 py-2 t-mono text-[11px] text-teal mt-3 break-all">{d.formula}</div>
            )}
          </div>
        )}
      </div>

      {/* Right sidebar note */}
      <div className="w-[200px] flex-shrink-0 border-l border-line2 bg-ink1 p-4 space-y-3">
        <div className="t-section">Context</div>
        <div className="text-t3 text-[11px] leading-relaxed">
          The model selected this candidate because its acquisition score ranked it among the top candidates in the unlabeled pool.
        </div>
        <hr className="rule2" />
        <div className="text-t3 text-[11px] leading-relaxed">
          <span className="text-amber font-medium">Simulated evaluation:</span> The true property value was revealed from the held-out dataset after selection.
        </div>
        <Link to={`/experiment/${id}/candidates`} className="btn btn-ghost w-full text-[12px] justify-center">
          ← All Candidates
        </Link>
      </div>
    </div>
  )
}
