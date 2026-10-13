import clsx from 'clsx'
import type { ReactNode } from 'react'

type A = 'teal' | 'vio' | 'amber' | 'ok' | 'err' | 'muted'

const COLORS: Record<A, string> = {
  teal: '#59D5CC', vio: '#A99BFF', amber: '#EDAB60',
  ok: '#41D98C', err: '#F06565', muted: '#9AA9BA',
}

interface Props {
  label: string
  value: string | number | null | undefined
  unit?: string; sub?: string; icon?: ReactNode
  accent?: A; className?: string
}

export default function MetricCard({ label, value, unit, sub, icon, accent = 'muted', className }: Props) {
  const color = COLORS[accent]
  const display = value != null ? (typeof value === 'number' ? value.toFixed(4) : String(value)) : '—'
  return (
    <div className={clsx('card p-3.5 flex flex-col gap-1', className)}>
      <div className="flex items-center justify-between">
        <span className="t-label">{label}</span>
        {icon && <span style={{ color }}>{icon}</span>}
      </div>
      <div className="flex items-baseline gap-1.5 mt-0.5">
        <span className="t-mono font-bold text-lg leading-none" style={{ color }}>
          {display}
        </span>
        {unit && value != null && <span className="text-t3 text-[11px]">{unit}</span>}
      </div>
      {sub && <div className="text-t3 text-[11px] mt-0.5">{sub}</div>}
    </div>
  )
}
