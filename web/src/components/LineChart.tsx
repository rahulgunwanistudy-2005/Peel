export interface Series {
  ink: 'raw' | 'peeled'
  label: string
  points: { x: number; y: number }[]
}

interface LineChartProps {
  series: Series[]
  xLabel: string
  yLabel: string
  title: string
  yDigits?: number
}

const W = 640
const H = 280
const PAD = { left: 74, right: 16, top: 16, bottom: 44 }

function niceMax(value: number): number {
  if (value <= 0) return 1
  const power = 10 ** Math.floor(Math.log10(value))
  const scaled = value / power
  const step = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10
  return step * power
}

/** A small two-series line chart drawn by hand in SVG. Magenta is the raw chain, cyan is Peel. */
export function LineChart({ series, xLabel, yLabel, title, yDigits = 2 }: LineChartProps) {
  const all = series.flatMap((s) => s.points)
  if (all.length === 0) return null
  const xMin = Math.min(...all.map((p) => p.x))
  const xMax = Math.max(...all.map((p) => p.x))
  const yMax = niceMax(Math.max(...all.map((p) => p.y)))
  const sx = (x: number) => PAD.left + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (W - PAD.left - PAD.right)
  const sy = (y: number) => H - PAD.bottom - (y / yMax) * (H - PAD.top - PAD.bottom)
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * yMax)
  const xs = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b)
  const xTicks = xs.length <= 12 ? xs : xs.filter((_, i) => i % Math.ceil(xs.length / 10) === 0)
  const summary = series
    .map((s) => `${s.label}: from ${s.points[0].y.toFixed(yDigits)} to ${s.points[s.points.length - 1].y.toFixed(yDigits)}`)
    .join('; ')

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}. ${summary}.`}>
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={sy(tick)} y2={sy(tick)} />
            <text x={PAD.left - 8} y={sy(tick) + 4} textAnchor="end">
              {tick.toFixed(yDigits)}
            </text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <text key={tick} x={sx(tick)} y={H - PAD.bottom + 18} textAnchor="middle">
            {tick}
          </text>
        ))}
        <path className="axis" d={`M${PAD.left} ${PAD.top}V${H - PAD.bottom}H${W - PAD.right}`} />
        <text x={(PAD.left + W - PAD.right) / 2} y={H - 6} textAnchor="middle">
          {xLabel}
        </text>
        <text transform={`translate(13 ${(PAD.top + H - PAD.bottom) / 2}) rotate(-90)`} textAnchor="middle">
          {yLabel}
        </text>
        {series.map((s) => (
          <g key={s.label}>
            <path className="series" data-ink={s.ink} d={s.points.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x)} ${sy(p.y)}`).join('')} />
            {s.points.map((p) => (
              <circle key={p.x} className="dot" data-ink={s.ink} cx={sx(p.x)} cy={sy(p.y)} r={3.5} />
            ))}
          </g>
        ))}
      </svg>
      <figcaption className="chart-legend">
        {series.map((s) => (
          <span key={s.label}>
            <i className="swatch" data-ink={s.ink === 'raw' ? 'revert' : 'keep'} /> {s.label}
          </span>
        ))}
      </figcaption>
    </figure>
  )
}
