import clsx from 'clsx'

interface Props { used: number; total: number; className?: string }

export default function BudgetMeter({ used, total, className }: Props) {
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0
  const color = pct < 55 ? '#59D5CC' : pct < 80 ? '#EDAB60' : '#F06565'

  return (
    <div className={clsx('card p-3.5', className)}>
      <div className="flex items-center justify-between mb-2">
        <span className="t-label">Experiment Budget</span>
        <span className="t-mono text-[11px] text-t2">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-[3px] bg-panel3 rounded-full overflow-hidden mb-3">
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <div className="flex justify-between text-[11px]">
        {[
          { l: 'Used',      v: used,         c: color },
          { l: 'Remaining', v: total - used, c: '#9AA9BA' },
          { l: 'Total',     v: total,        c: '#5C6E82' },
        ].map(({ l, v, c }) => (
          <div key={l} className="text-center">
            <div className="t-mono font-semibold text-sm" style={{ color: c }}>{v.toLocaleString()}</div>
            <div className="t-label mt-0.5">{l}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
