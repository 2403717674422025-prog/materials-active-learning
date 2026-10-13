import { useQuery } from '@tanstack/react-query'
import { Download, FileText } from 'lucide-react'
import { api } from '../services/api'

export default function ReportsPage() {
  const { data: experiments, isLoading } = useQuery({ queryKey: ['experiments-list'], queryFn: () => api.experiments.list(0, 50) })
  async function download(id: string, format: 'json' | 'csv') {
    const blob = await api.experiments.report(id, format)
    const url = URL.createObjectURL(blob as Blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `materials-al-${id.slice(0, 8)}.${format}`
    anchor.click()
    URL.revokeObjectURL(url)
  }
  return <div className="p-5 md:p-7 max-w-[1440px] mx-auto space-y-5">
    <header><div className="t-label t-mono mb-1">OUTPUTS / EXPORTS</div><h1 className="t-display workspace-heading">Reports</h1><p className="text-t3 mt-1">Download reports generated from saved experiment results.</p></header>
    <section className="card overflow-hidden">
      <div className="px-4 py-3 border-b border-line2 t-title">Experiment reports</div>
      {isLoading ? <div className="p-5 text-t3">Loading experiments…</div> : experiments?.length ? <div className="divide-y divide-line2">
        {experiments.map(exp => <div key={exp.id} className="px-4 py-3 flex items-center gap-4">
          <FileText className="w-4 h-4 text-vio" /><div className="flex-1 min-w-0"><div className="text-t1 font-medium truncate">{exp.name || exp.strategy.toUpperCase()}</div><div className="text-t3 text-[11px] t-mono">{exp.id} · {exp.current_iteration} iterations</div></div>
          <button className="btn btn-ghost text-[11px]" onClick={() => void download(exp.id, 'csv')}><Download className="w-3 h-3" />CSV</button><button className="btn btn-ghost text-[11px]" onClick={() => void download(exp.id, 'json')}><Download className="w-3 h-3" />JSON</button>
        </div>)}
      </div> : <div className="p-7 text-center text-t3 text-[12px]">Experiment reports will appear here after you create an experiment.</div>}
    </section>
  </div>
}
