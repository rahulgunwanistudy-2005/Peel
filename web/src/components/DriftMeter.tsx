import { fixed } from '../lib/format'
import { Num } from './Num'

/**
 * The drift score on a densitometer-style scale. Magenta is the ink for "changed without being
 * asked", so the filled part of the scale is magenta and nothing else on the ticket is.
 */
export function DriftMeter({ score }: { score: number }) {
  const clamped = Math.min(Math.max(score, 0), 100)
  return (
    <div className="drift">
      <div className="drift-figure">
        <span className="drift-number">
          <Num>{fixed(score, 1)}</Num>
        </span>
        <span className="drift-unit">
          drift score
          <span className="drift-of num"> out of 100</span>
        </span>
      </div>
      <div
        className="drift-scale"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(clamped * 10) / 10}
        aria-label="Drift score"
      >
        <div className="drift-fill" style={{ width: `${clamped}%` }} />
        <div className="drift-ticks" aria-hidden="true">
          {[0, 25, 50, 75, 100].map((tick) => (
            <span key={tick} className="num" style={{ left: `${tick}%` }}>
              {tick}
            </span>
          ))}
        </div>
      </div>
      <p className="drift-note">
        How much of the frame the editor changed without being asked, weighted by how visible each change is. 0 means
        nothing unrequested changed.
      </p>
    </div>
  )
}
