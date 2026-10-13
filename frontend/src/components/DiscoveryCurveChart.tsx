import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { StrategySeries } from '../types'

const STYLE: Record<string, { color: string; dash?: string; w: number }> = {
  random:      { color: '#3A4F64', dash: '4 3', w: 1.5 },
  uncertainty: { color: '#59D5CC', dash: '3 2', w: 1.5 },
  ei:          { color: '#A99BFF', w: 2 },
  ucb:         { color: '#EDAB60', w: 2 },
  puda:        { color: '#41D98C', w: 2.5 },
}

const CT = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card p-2 text-[11px] min-w-[130px]">
      <div className="t-label mb-1.5">{payload[0]?.payload?.budget_label ?? 'Cumulative candidates'}: {label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: p.color }} />
            <span className="text-t2">{p.name}</span>
          </div>
          <span className="t-mono font-semibold" style={{ color: p.color }}>
            {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
            {typeof p.payload?.[`${p.dataKey}_std`] === 'number'
              ? ` ±${p.payload[`${p.dataKey}_std`].toFixed(4)}`
              : ''}
            {typeof p.payload?.[`${p.dataKey}_n`] === 'number'
              ? ` (n=${p.payload[`${p.dataKey}_n`]})`
              : ''}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function DiscoveryCurveChart({ series, title, yKey = 'values', height = 220 }: {
  series: StrategySeries[]; title: string; yKey?: string; height?: number
}) {
  if (!series?.length) return (
    <div className="card flex items-center justify-center text-t3 text-[12px]" style={{ height }}>
      No data — run experiments with multiple strategies.
    </div>
  )
  const points = new Map<number, Record<string, number | string>>()
  for (const s of series) {
    const vals = (s as any)[yKey] as number[] | undefined
    for (let i = 0; i < Math.min(s.budget?.length ?? 0, vals?.length ?? 0); i++) {
      const x = s.budget[i]
      const row = points.get(x) ?? { budget: x }
      row[s.strategy] = vals![i]
      if (s.std?.[i] != null) row[`${s.strategy}_std`] = s.std[i]!
      if (s.n_runs?.[i] != null) row[`${s.strategy}_n`] = s.n_runs[i]!
      points.set(x, row)
    }
  }
  const data = [...points.values()].sort((a, b) => Number(a.budget) - Number(b.budget))
  const budgetLabel = series[0]?.budget_label ?? (
    series[0]?.budget_unit === 'training_candidates'
      ? 'Training Candidates'
      : 'Cumulative Candidates'
  )
  data.forEach(point => { point.budget_label = budgetLabel })
  return (
    <div className="card p-4">
      <div className="t-label mb-3">{title}</div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 12, left: -12, bottom: 14 }}>
          <CartesianGrid strokeDasharray="2 5" stroke="rgba(39,51,67,0.7)" />
          <XAxis dataKey="budget" type="number" domain={['dataMin', 'dataMax']} tick={{ fill: '#5C6E82', fontSize: 10 }}
            label={{ value: budgetLabel, position: 'insideBottom', offset: -8, fill: '#5C6E82', fontSize: 10 }} />
          <YAxis tick={{ fill: '#5C6E82', fontSize: 10 }} />
          <Tooltip content={<CT />} />
          {series.map(s => {
            const st = STYLE[s.strategy] ?? { color: s.color ?? '#888', w: 1.5 }
            return (
              <Line key={s.strategy} type="monotone" dataKey={s.strategy} name={s.label}
                stroke={st.color} strokeWidth={st.w} strokeDasharray={st.dash}
                dot={false} activeDot={{ r: 3, fill: st.color }} />
            )
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
