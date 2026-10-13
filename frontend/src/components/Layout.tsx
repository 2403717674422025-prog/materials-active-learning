import React, { useState } from 'react'
import { Outlet, NavLink, useParams, useLocation, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import {
  Atom, FlaskConical, Database, Activity, Table2,
  Clock, BarChart2, Award, ChevronRight,
  PanelLeft, PanelLeftClose, ExternalLink, Sparkles,
} from 'lucide-react'
import { api } from '../services/api'
import type { ExperimentStatusResponse } from '../types'

const NAV = [
  { to: '/',         icon: Atom,         label: 'Workspace', end: true },
  { to: '/dataset',  icon: Database,     label: 'Materials' },
  { to: '/history',  icon: FlaskConical, label: 'Experiments' },
  { to: '/benchmark',icon: BarChart2,    label: 'Benchmark Lab' },
  { to: '/reports',  icon: Clock,        label: 'Reports' },
]
function expNav(id: string) {
  return [
    { to: `/experiment/${id}`,            icon: Activity,  label: 'Console',    end: true },
    { to: `/experiment/${id}/candidates`, icon: Table2,    label: 'Candidates'           },
    { to: `/experiment/${id}/explainability`, icon: Sparkles, label: 'Explainability'       },
    { to: `/experiment/${id}/history`,    icon: Clock,     label: 'History'              },
    { to: `/experiment/${id}/compare`,    icon: BarChart2, label: 'Benchmark'            },
    { to: `/experiment/${id}/discovery`,  icon: Award,     label: 'Discovery'            },
  ]
}

function Logo({ collapsed }: { collapsed: boolean }) {
  return (
    <div className={clsx('flex items-center gap-2.5 px-4 py-[14px] border-b border-line2', collapsed && 'justify-center px-0')}>
      <svg width="22" height="22" viewBox="0 0 22 22" fill="none" className="flex-shrink-0">
        <circle cx="11" cy="11" r="3" fill="#59D5CC"/>
        <circle cx="4"  cy="5.5" r="2" fill="#A99BFF" opacity=".8"/>
        <circle cx="18" cy="5.5" r="2" fill="#A99BFF" opacity=".8"/>
        <circle cx="4"  cy="16.5" r="2" fill="#59D5CC" opacity=".55"/>
        <circle cx="18" cy="16.5" r="2" fill="#59D5CC" opacity=".55"/>
        <line x1="8.3" y1="9.3"  x2="5.6"  y2="6.9"  stroke="#273343" strokeWidth="1.2"/>
        <line x1="13.7" y1="9.3" x2="16.4" y2="6.9"  stroke="#273343" strokeWidth="1.2"/>
        <line x1="8.3" y1="12.7" x2="5.6"  y2="15.1" stroke="#273343" strokeWidth="1.2"/>
        <line x1="13.7" y1="12.7" x2="16.4" y2="15.1" stroke="#273343" strokeWidth="1.2"/>
      </svg>
      {!collapsed && (
        <div className="min-w-0">
          <div className="text-t1 font-semibold text-[13px] leading-none tracking-tight">Materials AL</div>
          <div className="text-t3 text-[10px] mt-0.5 leading-none">Discovery Intelligence</div>
        </div>
      )}
    </div>
  )
}

function ApiDot() {
  const { data, isError } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const response = await fetch('/api/health')
      if (!response.ok) throw new Error('API unavailable')
      return response.json()
    },
    refetchInterval: 30_000, retry: false,
  })
  const ok = !isError && data?.status === 'ok'
  return (
    <div className="flex items-center gap-1.5 px-4 py-2">
      <span className={clsx('dot', ok ? 'dot-live' : 'dot-err')} />
      <span className="text-[11px] text-t3">{ok ? `API ${data?.version ? `v${data.version}` : 'connected'}` : 'API offline'}</span>
    </div>
  )
}

function Crumb() {
  const loc = useLocation()
  const { id, candidateId } = useParams<{ id?: string; candidateId?: string }>()
  const seg = loc.pathname.split('/').filter(Boolean)
  const parts: { label: string; to?: string }[] = []

  if (!seg.length) { parts.push({ label: 'Workspace' }); }
  else if (seg[0] === 'dataset') parts.push({ label: 'Materials' })
  else if (seg[0] === 'setup')   parts.push({ label: 'New Experiment' })
  else if (seg[0] === 'benchmark') parts.push({ label: 'Benchmark Lab' })
  else if (seg[0] === 'reports') parts.push({ label: 'Reports' })
  else if (seg[0] === 'history') parts.push({ label: 'Experiments' })
  else if (seg[0] === 'experiment' && id) {
        parts.push({ label: id.slice(0,8) + '…', to: `/experiment/${id}` })
    if (seg[2] === 'candidates' && candidateId) {
      parts.push({ label: 'Candidates', to: `/experiment/${id}/candidates` })
      parts.push({ label: candidateId.slice(0,14) })
    } else if (seg[2] === 'candidates') parts.push({ label: 'Candidates' })
    else if (seg[2] === 'explainability') parts.push({ label: 'Explainability' })
    else if (seg[2] === 'history')      parts.push({ label: 'History' })
    else if (seg[2] === 'compare')      parts.push({ label: 'Benchmark' })
    else if (seg[2] === 'discovery')    parts.push({ label: 'Discovery' })
    else parts.push({ label: 'Console' })
  }

  return (
    <nav className="flex items-center gap-1 text-[12px]">
      {parts.map((p, i) => (
        <span key={i} className="flex items-center gap-1">
          {i > 0 && <ChevronRight className="w-3 h-3 text-t4" />}
          {p.to
            ? <Link to={p.to} className="text-t3 hover:text-t1 transition-colors t-mono">{p.label}</Link>
            : <span className={clsx('t-mono', i === parts.length - 1 ? 'text-t2' : 'text-t3')}>{p.label}</span>
          }
        </span>
      ))}
    </nav>
  )
}

export default function Layout() {
  const { id } = useParams<{ id: string }>()
  const [col, setCol] = useState(false)
  const nav2 = id ? expNav(id) : []

  const { data: st } = useQuery<ExperimentStatusResponse>({
    queryKey: ['experiment-status', id],
    queryFn: () => api.experiments.status(id!),
    enabled: !!id, refetchInterval: 8_000,
  })
  const { data: experimentContext } = useQuery({
    queryKey: ['experiment-context', id],
    queryFn: async () => (await api.experiments.list(0, 100)).find(e => e.id === id),
    enabled: !!id,
    staleTime: 30_000,
  })
  const { data: datasets } = useQuery({
    queryKey: ['datasets'], queryFn: api.datasets.list, enabled: !!id, staleTime: 30_000,
  })
  const activeDataset = datasets?.find(d => d.id === experimentContext?.dataset_id)

  function Item({ to, icon: Icon, label, end }: { to: string; icon: React.ElementType; label: string; end?: boolean }) {
    return (
      <NavLink to={to} end={end} className={({ isActive }) => clsx(
        'flex items-center gap-2.5 px-3 py-[7px] rounded-[4px] transition-all text-[13px]',
        col && 'justify-center px-0 py-2.5',
        isActive
          ? 'bg-[rgba(89,213,204,0.08)] text-teal border-l-2 border-teal pl-[10px]'
          : 'text-t3 hover:text-t1 hover:bg-white/[0.03] border-l-2 border-transparent',
      )}>
        <Icon style={{ width: 15, height: 15, flexShrink: 0 }} />
        {!col && <span className="font-medium truncate">{label}</span>}
      </NavLink>
    )
  }

  return (
      <div className="app-shell flex h-screen bg-ink overflow-hidden">
      {/* ── Sidebar ─────────────────────────────────── */}
      <aside className={clsx(
        'app-sidebar flex-shrink-0 flex flex-col bg-ink1 border-r border-line2 transition-all duration-150 relative',
        col ? 'w-[44px]' : 'w-[192px]',
      )}>
        <Logo collapsed={col} />

        <button
          onClick={() => setCol(v => !v)}
          className="absolute -right-[11px] top-[52px] w-[22px] h-[22px] rounded-full bg-panel border border-line flex items-center justify-center text-t3 hover:text-t1 hover:border-teal/40 transition-all z-20"
        >
          {col ? <PanelLeft className="w-3 h-3" /> : <PanelLeftClose className="w-3 h-3" />}
        </button>

        <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-0.5">
          {!col && <div className="px-2 pb-1"><span className="t-section text-[10px]">Platform</span></div>}
          {NAV.map(n => <Item key={n.to} {...n} />)}

          {nav2.length > 0 && (
            <>
              <div className="pt-3 pb-1 px-2">
                {!col ? <span className="t-section text-[10px]">Experiment</span> : <hr className="rule2" />}
              </div>
              {nav2.map(n => <Item key={n.to} {...n} />)}
            </>
          )}
        </nav>

        <div className="border-t border-line2">
          <ApiDot />
          {!col && (
            <a href="/api/docs" target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 text-[12px] text-t3 hover:text-t1 transition-colors">
              <ExternalLink className="w-3.5 h-3.5" />API Docs
            </a>
          )}
        </div>
      </aside>

      {/* ── Main ─────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Top bar — thin, functional */}
        <header className="app-topbar h-[40px] flex items-center px-5 gap-4 border-b border-line2 bg-ink1/80 backdrop-blur flex-shrink-0">
          <Crumb />
          <div className="flex-1" />

          {experimentContext && (
            <div className="hidden lg:flex items-center gap-2 text-[11px] min-w-0">
              <span className="text-t3 truncate max-w-40">{experimentContext.name || experimentContext.strategy.toUpperCase()}</span>
              <span className="text-line">·</span>
              <Link to="/dataset" className="text-t3 hover:text-teal truncate max-w-40" title="Open active dataset">{activeDataset?.name ?? 'Dataset unavailable'}</Link>
              <span className="chip chip-vio">{experimentContext.strategy.toUpperCase()}</span>
            </div>
          )}

          {/* Live experiment budget indicator */}
          {st && (
            <div className="flex items-center gap-2 text-[12px]">
              <span className={clsx('dot', ['initialized', 'running'].includes(st.status) ? 'dot-live' : st.status === 'completed' ? 'dot-ok' : 'dot-idle')} />
              <span className="text-t3">Budget</span>
              <span className="t-mono text-teal font-semibold">{st.budget_used}</span>
              <span className="text-t4">/</span>
              <span className="t-mono text-t3">{st.budget_total}</span>
            </div>
          )}

          {st && (
            <div className="h-4 border-r border-line2" />
          )}

          {st && ['initialized', 'running'].includes(st.status) ? (
            <Link to={`/experiment/${id}`} className="btn btn-teal text-[12px] py-1 px-3"><Activity style={{ width: 13, height: 13 }} />Run Iteration</Link>
          ) : <Link to="/setup" className="btn btn-teal text-[12px] py-1 px-3"><FlaskConical style={{ width: 13, height: 13 }} />New Experiment</Link>}
        </header>

        <main className="flex-1 overflow-auto animate-in">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
