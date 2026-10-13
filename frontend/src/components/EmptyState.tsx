import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

export default function EmptyState({ icon, title, description, action }: {
  icon?: ReactNode; title: string; description?: string
  action?: { label: string; to?: string; onClick?: () => void }
}) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-6 text-center">
      {icon && (
        <div className="w-10 h-10 rounded-md bg-panel2 border border-line flex items-center justify-center text-t3 mb-3">
          {icon}
        </div>
      )}
      <div className="text-t2 text-[13px] font-medium mb-1">{title}</div>
      {description && <p className="text-t3 text-[12px] max-w-xs leading-relaxed mb-4">{description}</p>}
      {action && (
        action.to
          ? <Link to={action.to} className="btn btn-teal text-[12px]">{action.label}</Link>
          : <button className="btn btn-teal text-[12px]" onClick={action.onClick}>{action.label}</button>
      )}
    </div>
  )
}
