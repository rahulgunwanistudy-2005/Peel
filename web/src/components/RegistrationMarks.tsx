export type MarkPhase = 'out' | 'registering' | 'registered'

const CORNERS = ['tl', 'tr', 'bl', 'br'] as const

function Crosshair({ ink }: { ink: 'cyan' | 'magenta' | 'key' }) {
  return (
    <svg className={`mark-ink mark-${ink}`} viewBox="0 0 28 28" aria-hidden="true">
      <circle cx="14" cy="14" r="7" />
      <path d="M14 0v28M0 14h28" />
    </svg>
  )
}

/**
 * Printer's registration marks at the corners of the image. Out of register they show as separate
 * cyan and magenta crosshairs; in register the two sit exactly on top of each other and read as one
 * black mark. The slide between the two states is the interface's one motion moment.
 */
export function RegistrationMarks({ phase }: { phase: MarkPhase }) {
  return (
    <div className="marks" data-phase={phase} aria-hidden="true">
      {CORNERS.map((corner) => (
        <div key={corner} className={`mark mark-${corner}`}>
          <Crosshair ink="cyan" />
          <Crosshair ink="magenta" />
          <Crosshair ink="key" />
        </div>
      ))}
    </div>
  )
}
