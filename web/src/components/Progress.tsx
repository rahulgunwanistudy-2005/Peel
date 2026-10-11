import type { ProgressEvent } from '../api/client'

const ORDER = ['decode', 'register', 'change', 'intent', 'faces', 'classify', 'composite', 'ledger']
const SHORT: Record<string, string> = {
  decode: 'Read',
  register: 'Register',
  change: 'Measure',
  intent: 'Locate',
  faces: 'Faces',
  classify: 'Sort',
  composite: 'Composite',
  ledger: 'Ledger',
}

/** The pipeline's real stages, as they arrive over the event stream. */
export function Progress({ event }: { event: ProgressEvent }) {
  const current = ORDER.indexOf(event.stage)
  return (
    <div className="progress" role="status" aria-live="polite">
      <p className="progress-label">{event.label || 'Working'}</p>
      <ol className="progress-steps" aria-hidden="true">
        {ORDER.map((stage, index) => (
          <li key={stage} data-state={index < current ? 'done' : index === current ? 'now' : 'todo'}>
            {SHORT[stage]}
          </li>
        ))}
      </ol>
    </div>
  )
}
