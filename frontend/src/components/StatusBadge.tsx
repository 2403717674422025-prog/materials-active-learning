import clsx from 'clsx'

const CFG: Record<string, { cls: string; dot: string; label?: string }> = {
  running:     { cls: 'chip-teal',  dot: 'dot-live',  label: 'Running'     },
  initialized: { cls: 'chip-vio',   dot: 'dot-vio',   label: 'Ready'       },
  completed:   { cls: 'chip-ok',    dot: 'dot-ok',    label: 'Completed'   },
  ready:       { cls: 'chip-ok',    dot: 'dot-ok',    label: 'Ready'       },
  pending:     { cls: 'chip-amber', dot: 'dot-amber', label: 'Loading'     },
  created:     { cls: 'chip-idle',  dot: 'dot-idle',  label: 'Created'     },
  error:       { cls: 'chip-err',   dot: 'dot-err',   label: 'Error'       },
}

export default function StatusBadge({ status }: { status: string }) {
  const c = CFG[status] ?? { cls: 'chip-idle', dot: 'dot-idle', label: status }
  return (
    <span className={clsx('chip', c.cls)}>
      <span className={clsx('dot', c.dot)} />
      {c.label ?? status}
    </span>
  )
}
