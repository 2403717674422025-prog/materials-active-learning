import clsx from 'clsx'
export const Sk = ({ className }: { className?: string }) => <div className={clsx('sk', className)} />
export const SkCard = () => (
  <div className="card p-3.5 space-y-2">
    <div className="sk h-2.5 w-20" /><div className="sk h-6 w-28" /><div className="sk h-2 w-16" />
  </div>
)
export const SkTable = ({ rows = 5 }: { rows?: number }) => (
  <div className="card overflow-hidden">
    <div className="bg-panel2 px-4 py-2 flex gap-4 border-b border-line2">
      {['w-20','w-14','w-14','w-12','w-12'].map((w,i) => <div key={i} className={clsx('sk h-2',w)} />)}
    </div>
    {Array.from({length:rows}).map((_,i) => (
      <div key={i} className="px-4 py-2.5 flex gap-4 border-b border-line2 last:border-0">
        {['w-24','w-16','w-16','w-14','w-14'].map((w,j) => <div key={j} className={clsx('sk h-2',w)} />)}
      </div>
    ))}
  </div>
)
