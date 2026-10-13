import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { Search, Sparkles } from 'lucide-react'
import { api } from '../services/api'
import type { ShapBeeswarmPoint } from '../types'

const tick = { fill: '#7E91A8', fontSize: 10 }

function errorMessage(error: unknown, fallback: string) {
  const detail = (error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
  return typeof detail === 'string' ? detail : fallback
}

function scaledColors(points: ShapBeeswarmPoint[]) {
  const ranges = new Map<string, [number, number]>()
  for (const point of points) {
    const [lo, hi] = ranges.get(point.feature) ?? [point.feature_value, point.feature_value]
    ranges.set(point.feature, [Math.min(lo, point.feature_value), Math.max(hi, point.feature_value)])
  }
  return points.map(point => {
    const [lo, hi] = ranges.get(point.feature)!
    return hi === lo ? 0.5 : (point.feature_value - lo) / (hi - lo)
  })
}

function featureColor(value: number) {
  const t = Math.min(1, Math.max(0, value))
  const low = [169, 155, 255]
  const high = [89, 213, 204]
  const rgb = low.map((channel, index) => Math.round(channel + (high[index] - channel) * t))
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}

export default function ExplainabilityPage() {
  const { id } = useParams<{ id: string }>()
  const [candidateSearch, setCandidateSearch] = useState('')
  const [candidateId, setCandidateId] = useState('')
  const [requestedCandidate, setRequestedCandidate] = useState('')
  const [feature, setFeature] = useState('')

  const global = useQuery({
    queryKey: ['shap-global', id],
    queryFn: () => api.experiments.explainability.global(id!),
    enabled: !!id,
    retry: false,
  })
  const matches = useQuery({
    queryKey: ['shap-candidates', id, candidateSearch],
    queryFn: () => api.experiments.explainability.candidates(id!, candidateSearch),
    enabled: !!id && candidateSearch.trim().length > 0,
    staleTime: 20_000,
  })
  const candidate = useQuery({
    queryKey: ['shap-candidate', id, requestedCandidate],
    queryFn: () => api.experiments.explainability.candidate(id!, requestedCandidate),
    enabled: !!id && !!requestedCandidate,
    retry: false,
  })
  const dependence = useQuery({
    queryKey: ['shap-dependence', id, feature],
    queryFn: () => api.experiments.explainability.dependence(id!, feature),
    enabled: !!id && !!feature,
    retry: false,
  })

  const importance = global.data?.feature_importance ?? []
  const displayedImportance = importance.slice(0, 20)
  useEffect(() => {
    if (importance.length && !importance.some(item => item.feature === feature)) {
      setFeature(importance[0].feature)
    }
  }, [importance, feature])

  const beeswarmColors = useMemo(
    () => scaledColors(global.data?.beeswarm_points ?? []),
    [global.data?.beeswarm_points],
  )
  const featureIndex = useMemo(
    () => new Map(displayedImportance.map((item, index) => [item.feature, index])),
    [displayedImportance],
  )
  const beeswarmData = (global.data?.beeswarm_points ?? []).map((point, index) => ({
    ...point,
    y: (featureIndex.get(point.feature) ?? 0) + ((index % 7) - 3) * 0.055,
    color: featureColor(beeswarmColors[index]),
  }))
  const waterfall = candidate.data
    ? (() => {
        const top = candidate.data.contributions.slice(0, 14)
        const remainder = candidate.data.contributions.slice(14)
        const rows: { label: string; offset: number; bar: number; color: string }[] = [
          { label: 'Baseline', offset: 0, bar: candidate.data.baseline, color: '#A99BFF' },
        ]
        let cumulative = candidate.data.baseline
        const addContribution = (label: string, amount: number) => {
          const next = cumulative + amount
          rows.push({
            label, offset: Math.min(cumulative, next), bar: Math.abs(amount),
            color: amount >= 0 ? '#59D5CC' : '#F06565',
          })
          cumulative = next
        }
        for (const item of top) addContribution(item.feature, item.shap_value)
        if (remainder.length) {
          addContribution('Other features', remainder.reduce((sum, item) => sum + item.shap_value, 0))
        }
        rows.push({ label: 'Prediction', offset: 0, bar: candidate.data.predicted_value, color: '#A99BFF' })
        return rows
      })()
    : null

  return (
    <div className="px-5 py-5 lg:px-7 space-y-5 overflow-auto h-full">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="t-label text-teal mb-1">Model interpretation</div>
          <h1 className="t-display flex items-center gap-2"><Sparkles className="w-5 h-5 text-teal" />Explainability</h1>
          <p className="text-t3 text-[12px] mt-2 max-w-3xl">
            SHAP values explain this experiment’s fitted {global.data?.model_type ?? 'surrogate'} predictions. Positive contributions
            raise a prediction above the baseline; negative contributions lower it. These are model
            attributions, not causal effects.
          </p>
        </div>
        {global.data && <span className="chip chip-vio">{global.data.model_type} · iteration {global.data.model_iteration}</span>}
      </header>

      {global.isLoading && <div className="card p-5 text-t3">Computing SHAP values for a reproducible pool sample…</div>}
      {global.isError && <div className="card p-5 text-amber text-[12px]">{errorMessage(global.error, 'SHAP explanations are unavailable for this experiment.')}</div>}

      {global.data && (
        <>
          <div className="grid xl:grid-cols-2 gap-4">
            <section className="card p-4 min-w-0">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div><div className="t-label">Top 20 global feature importance</div><div className="text-t3 text-[11px] mt-1">Mean absolute SHAP value · {global.data.sample_size} pool candidates · {global.data.total_features} named features ranked</div></div>
                <span className="t-mono text-[11px] text-t3">Baseline {global.data.baseline.toFixed(4)}</span>
              </div>
              <ResponsiveContainer width="100%" height={Math.max(300, displayedImportance.length * 25)}>
                <BarChart data={displayedImportance} layout="vertical" margin={{ top: 8, right: 18, bottom: 12, left: 8 }}>
                  <CartesianGrid stroke="#1A2C40" horizontal={false} />
                  <XAxis type="number" tick={tick} axisLine={{ stroke: '#273343' }} tickLine={false} />
                  <YAxis type="category" dataKey="feature" width={160} tick={tick} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#0A1523', border: '1px solid #273343', borderRadius: 5 }} formatter={(value) => [Number(value).toPrecision(4), 'Mean |SHAP|']} />
                  <Bar dataKey="mean_abs_shap" fill="#59D5CC" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="card p-4 min-w-0">
              <div className="t-label mb-1">SHAP beeswarm</div>
              <div className="text-t3 text-[11px]">Each point is a pool candidate; color shows relative feature value within that feature.</div>
              <ResponsiveContainer width="100%" height={Math.max(300, displayedImportance.length * 25)}>
                <ScatterChart margin={{ top: 8, right: 12, bottom: 16, left: 8 }}>
                  <CartesianGrid stroke="#1A2C40" />
                  <XAxis type="number" dataKey="shap_value" name="SHAP value" tick={tick} axisLine={{ stroke: '#273343' }} tickLine={false} label={{ value: 'SHAP value (effect on prediction)', position: 'insideBottom', offset: -10, fill: '#7E91A8', fontSize: 10 }} />
                  <YAxis type="number" dataKey="y" domain={[-0.5, displayedImportance.length - 0.5]} ticks={displayedImportance.map((_, index) => index)} tickFormatter={value => displayedImportance[Number(value)]?.feature ?? ''} tick={tick} width={160} reversed axisLine={false} tickLine={false} />
                  <ZAxis range={[35, 35]} />
                  <ReferenceLine x={0} stroke="#64748B" strokeDasharray="3 3" />
                  <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="card p-2 text-[10px]"><div className="text-t1">{payload[0].payload.feature}</div><div className="text-t3">{payload[0].payload.candidate_id}</div><div>Feature value: {Number(payload[0].payload.feature_value).toPrecision(4)}</div><div>SHAP: {Number(payload[0].payload.shap_value).toPrecision(4)}</div></div> : null} />
                  <Scatter data={beeswarmData} isAnimationActive={false}>
                    {beeswarmData.map((point, index) => <Cell key={`${point.candidate_id}-${point.feature}-${index}`} fill={point.color} fillOpacity={0.78} />)}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </section>
          </div>

          <section className="card p-4">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
              <div><div className="t-label">Individual candidate explanation</div><div className="text-t3 text-[11px] mt-1">Choose a candidate from this experiment’s active-learning pool.</div></div>
              <div className="flex gap-2">
                <div className="relative">
                  <Search className="absolute left-2 top-2 w-3.5 h-3.5 text-t4" />
                  <input className="input text-[12px] pl-7 py-1.5 w-64" value={candidateId}
                    onChange={event => { setCandidateId(event.target.value); setCandidateSearch(event.target.value); setRequestedCandidate('') }}
                    list="shap-candidate-options" placeholder="Search candidate ID…" />
                  <datalist id="shap-candidate-options">{(matches.data?.candidates ?? []).map(item => <option key={item} value={item} />)}</datalist>
                </div>
                <button className="btn btn-teal text-[12px] py-1.5 px-3" disabled={!candidateId.trim()} onClick={() => setRequestedCandidate(candidateId.trim())}>Explain</button>
              </div>
            </div>
            {candidate.isFetching && <div className="py-6 text-t3 text-[12px]">Computing candidate attribution…</div>}
            {candidate.isError && requestedCandidate && <div className="py-4 text-amber text-[12px]">{errorMessage(candidate.error, 'Could not explain this candidate.')}</div>}
            {candidate.data && waterfall && (
              <div className="grid xl:grid-cols-[minmax(0,1.4fr)_minmax(240px,0.6fr)] gap-4 items-start">
                <div className="min-w-0">
                  <div className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] mb-2">
                    <span className="text-t3">Prediction <b className="t-mono text-t1">{candidate.data.predicted_value.toFixed(5)}</b></span>
                    <span className="text-t3">SHAP baseline <b className="t-mono text-t1">{candidate.data.baseline.toFixed(5)}</b></span>
                    <span className="text-t3">Candidate <b className="t-mono text-teal">{candidate.data.candidate_id}</b></span>
                  </div>
                  <ResponsiveContainer width="100%" height={380}>
                    <BarChart data={waterfall} margin={{ top: 12, right: 20, bottom: 82, left: 8 }}>
                      <CartesianGrid stroke="#1A2C40" vertical={false} />
                      <XAxis dataKey="label" interval={0} angle={-40} textAnchor="end" height={92} tick={tick} axisLine={{ stroke: '#273343' }} tickLine={false} />
                      <YAxis tick={tick} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: '#0A1523', border: '1px solid #273343', borderRadius: 5 }} formatter={(value, name, item) => name === 'bar' ? [Number(value).toPrecision(5), item.payload.label] : [null, null]} />
                      <Bar dataKey="offset" stackId="waterfall" fill="transparent" isAnimationActive={false} />
                      <Bar dataKey="bar" stackId="waterfall" isAnimationActive={false}>
                        {waterfall.map(item => <Cell key={item.label} fill={item.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-4 text-[12px]">
                  <p className="text-t2 leading-relaxed">{candidate.data.interpretation}</p>
                  <div><div className="t-label text-teal mb-1">Largest positive contributions</div>{candidate.data.largest_positive.length ? candidate.data.largest_positive.slice(0, 3).map(item => <div className="flex justify-between gap-3 py-1" key={item.feature}><span className="text-t2 truncate">{item.feature}</span><span className="t-mono text-teal">+{item.shap_value.toFixed(4)}</span></div>) : <span className="text-t3">None</span>}</div>
                  <div><div className="t-label text-red-300 mb-1">Largest negative contributions</div>{candidate.data.largest_negative.length ? candidate.data.largest_negative.slice(0, 3).map(item => <div className="flex justify-between gap-3 py-1" key={item.feature}><span className="text-t2 truncate">{item.feature}</span><span className="t-mono text-red-300">{item.shap_value.toFixed(4)}</span></div>) : <span className="text-t3">None</span>}</div>
                </div>
              </div>
            )}
          </section>

          <section className="card p-4">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
              <div><div className="t-label">Feature dependence</div><div className="text-t3 text-[11px] mt-1">Pool feature values against their SHAP contributions.</div></div>
              <select className="input text-[12px] py-1.5 max-w-80" value={feature} onChange={event => setFeature(event.target.value)}>
                {importance.map(item => <option key={item.feature} value={item.feature}>{item.feature}</option>)}
              </select>
            </div>
            {dependence.data && (
              <ResponsiveContainer width="100%" height={330}>
                <ScatterChart margin={{ top: 8, right: 20, bottom: 24, left: 8 }}>
                  <CartesianGrid stroke="#1A2C40" />
                  <XAxis type="number" dataKey="feature_value" name={feature} tick={tick} axisLine={{ stroke: '#273343' }} tickLine={false} />
                  <YAxis type="number" dataKey="shap_value" name="SHAP value" tick={tick} axisLine={false} tickLine={false} />
                  <ReferenceLine y={0} stroke="#64748B" strokeDasharray="3 3" />
                  <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="card p-2 text-[10px]"><div className="text-t3">{payload[0].payload.candidate_id}</div><div>Feature value: {Number(payload[0].payload.feature_value).toPrecision(4)}</div><div>SHAP: {Number(payload[0].payload.shap_value).toPrecision(4)}</div></div> : null} />
                  <Scatter data={dependence.data.points} fill="#59D5CC" isAnimationActive={false} />
                </ScatterChart>
              </ResponsiveContainer>
            )}
            {dependence.isError && <div className="py-5 text-amber text-[12px]">{errorMessage(dependence.error, 'Could not compute feature dependence.')}</div>}
          </section>

          <p className="text-t4 text-[10px]">Global and dependence summaries use a deterministic sample of active-learning pool features. {global.data.reference_method} Held-out rows and all target values are excluded. {global.data.feature_value_source}</p>
        </>
      )}
    </div>
  )
}
