export function fixed(value: number, digits = 1): string {
  return value.toFixed(digits)
}

export function percent(value: number, digits = 1): string {
  return `${value.toFixed(digits)}%`
}

export function seconds(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`
}

export function bytes(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`
  return `${Math.max(1, Math.round(n / 1e3))} kB`
}
