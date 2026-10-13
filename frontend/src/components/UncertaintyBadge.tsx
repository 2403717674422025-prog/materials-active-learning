import clsx from 'clsx'

const CFG = {
  Low:    { cls: 'chip-ok',    dot: 'dot-ok'    },
  Medium: { cls: 'chip-amber', dot: 'dot-amber'  },
  High:   { cls: 'chip-vio',   dot: 'dot-vio'    },
}

export default function UncertaintyBadge({ label, value, showValue = false }: {
  label: string; value?: number; showValue?: boolean
}) {
  const c = CFG[label as keyof typeof CFG] ?? { cls: 'chip-idle', dot: 'dot-idle' }
  return (
    <span className={clsx('chip', c.cls)}>
      <span className={clsx('dot flex-shrink-0', c.dot)} />
      {label}
      {showValue && value != null && <span className="t-mono opacity-60 ml-0.5">±{value.toFixed(3)}</span>}
    </span>
  )
}
