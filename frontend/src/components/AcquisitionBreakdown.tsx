interface Props {
  alpha: number; beta: number; gamma: number
  predNorm: number; uncNorm: number; divNorm: number
  finalScore: number; strategy?: string
}
export default function AcquisitionBreakdown({
  alpha, beta, gamma, predNorm, uncNorm, divNorm, finalScore, strategy = 'PUDA',
}: Props) {
  const bars = [
    { k: 'Performance', v: alpha * predNorm, w: predNorm, c: '#59D5CC' },
    { k: 'Uncertainty', v: beta  * uncNorm,  w: uncNorm,  c: '#A99BFF' },
    { k: 'Diversity',   v: gamma * divNorm,  w: divNorm,  c: '#EDAB60' },
  ]
  const total = bars.reduce((s, b) => s + b.v, 0) || 1

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="t-label">{strategy} Score</span>
        <span className="t-mono text-teal font-bold">{finalScore.toFixed(4)}</span>
      </div>

      {/* Stacked bar */}
      <div className="h-2 rounded-sm overflow-hidden flex gap-px bg-panel2">
        {bars.map(b => (
          <div key={b.k} style={{ width: `${(b.v/total)*100}%`, background: b.c, opacity: 0.8 }} />
        ))}
      </div>

      {/* Rows */}
      <div className="space-y-1.5">
        {bars.map(b => (
          <div key={b.k} className="flex items-center gap-2.5">
            <div className="w-2 h-2 rounded-sm flex-shrink-0" style={{ background: b.c }} />
            <span className="text-t3 text-[11px] w-20">{b.k}</span>
            <div className="flex-1 h-1 bg-panel3 rounded-sm overflow-hidden">
              <div style={{ width: `${b.w*100}%`, background: b.c, opacity: 0.6 }} className="h-full" />
            </div>
            <span className="t-mono text-[11px] text-t2 w-14 text-right">{b.v.toFixed(4)}</span>
          </div>
        ))}
      </div>

      {/* Formula */}
      <div className="inset px-3 py-2 t-mono text-[11px] text-t2 leading-relaxed">
        <span style={{ color: '#59D5CC' }}>{alpha}×{predNorm.toFixed(3)}</span>
        {' + '}
        <span style={{ color: '#A99BFF' }}>{beta}×{uncNorm.toFixed(3)}</span>
        {' + '}
        <span style={{ color: '#EDAB60' }}>{gamma}×{divNorm.toFixed(3)}</span>
        {' = '}
        <span className="text-teal font-semibold">{finalScore.toFixed(4)}</span>
      </div>
    </div>
  )
}



