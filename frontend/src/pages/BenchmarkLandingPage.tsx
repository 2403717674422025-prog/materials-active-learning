import { useQuery } from '@tanstack/react-query'
import { ArrowRight, BarChart3, FlaskConical } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'

export default function BenchmarkLandingPage() {
  const { data: experiments, isLoading } = useQuery({ queryKey: ['experiments-list'], queryFn: () => api.experiments.list(0, 50) })
  return <div className="p-5 md:p-7 max-w-[1440px] mx-auto space-y-5">
    <header className="flex items-end justify-between gap-4">
      <div><div className="t-label t-mono mb-1">EVALUATION / STRATEGY COMPARISON</div><h1 className="t-display workspace-heading">Benchmark Lab</h1><p className="text-t3 mt-1">Run matched comparisons using an experiment’s dataset, initialization, batch size, and budget.</p></div>
      <BarChart3 className="w-5 h-5 text-teal" />
    </header>
    <section className="card overflow-hidden">
      <div className="px-4 py-3 border-b border-line2 flex items-center justify-between"><div className="t-title">Choose a benchmark configuration</div><span className="text-t3 text-[11px]">Measured results only</span></div>
      {isLoading ? <div className="p-5 text-t3">Loading experiments…</div> : experiments?.length ? <div className="divide-y divide-line2">
        {experiments.map(exp => <div key={exp.id} className="px-4 py-3 flex items-center gap-4 hover:bg-white/[0.02]">
          <div className="w-8 h-8 rounded bg-panel2 flex items-center justify-center"><FlaskConical className="w-4 h-4 text-teal" /></div>
          <div className="min-w-0 flex-1"><div className="text-t1 font-medium truncate">{exp.name || `${exp.strategy.toUpperCase()} experiment`}</div><div className="text-t3 text-[11px] t-mono">{exp.model_type.toUpperCase()} · {exp.current_budget_used}/{exp.total_budget} observations · {exp.status}</div></div>
          <Link to={`/experiment/${exp.id}/compare`} className="btn btn-ghost text-[11px]">Open comparison <ArrowRight className="w-3 h-3" /></Link>
        </div>)}
      </div> : <div className="p-7 text-center"><div className="text-t2 font-medium">No experiment configurations yet</div><p className="text-t3 text-[12px] mt-1">Create and initialize an experiment to use its settings for a matched strategy benchmark.</p><Link className="btn btn-teal mt-4" to="/setup">New Experiment</Link></div>}
    </section>
  </div>
}
