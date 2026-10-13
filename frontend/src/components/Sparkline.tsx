export default function Sparkline({ data, color = '#59D5CC', height = 32, width = 80, inverted = false }: {
  data: number[]; color?: string; height?: number; width?: number; inverted?: boolean
}) {
  if (!data || data.length < 2) return null
  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width
    const y = inverted ? ((v - min) / range) * (height - 4) + 2
                       : height - ((v - min) / range) * (height - 4) - 2
    return `${x},${y}`
  }).join(' ')
  const last = data[data.length - 1]
  const lx = width
  const ly = inverted ? ((last - min) / range) * (height - 4) + 2
                      : height - ((last - min) / range) * (height - 4) - 2
  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5"
        strokeLinejoin="round" strokeLinecap="round" opacity="0.65" />
      <circle cx={lx} cy={ly} r="2.5" fill={color} />
    </svg>
  )
}
