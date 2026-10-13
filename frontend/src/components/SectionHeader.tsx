import type { ReactNode } from 'react'
export default function SectionHeader({ title, sub, action }: { title: string; sub?: string; action?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 mb-4">
      <div>
        <div className="t-title text-[13px]">{title}</div>
        {sub && <div className="text-t3 text-[11px] mt-0.5">{sub}</div>}
      </div>
      {action}
    </div>
  )
}
