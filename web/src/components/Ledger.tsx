import { downloadUrl, imageUrl, type Decision, type PeelResult, type Region } from '../api/client'
import { fixed, percent, seconds } from '../lib/format'
import { DriftMeter } from './DriftMeter'
import { Num } from './Num'

const TYPE_LABELS: Record<Region['type'], string> = {
  color_cast: 'Colour',
  texture: 'Texture',
  structure: 'Structure',
  face_identity: 'Face identity',
  text: 'Text',
  object: 'Content',
}

const THUMB = 56

function ink(region: Region): 'keep' | 'revert' | 'review' {
  if (region.classification === 'ambiguous' && !region.overridden) return 'review'
  return region.decision === 'keep' ? 'keep' : 'revert'
}

function inkWord(region: Region): string {
  const which = ink(region)
  if (which === 'review') return 'Kept, please review'
  if (which === 'keep') return region.overridden ? 'Kept by you' : 'Kept'
  return region.overridden ? 'Peeled by you' : 'Peeled'
}

/** A crop of one image around a region, drawn with CSS so no canvas work is needed. */
function Crop({ url, region, width, height, label }: { url: string; region: Region; width: number; height: number; label: string }) {
  const pad = Math.max(region.bbox.w, region.bbox.h) * 0.15
  const side = Math.max(region.bbox.w, region.bbox.h) + 2 * pad
  const cx = region.bbox.x + region.bbox.w / 2
  const cy = region.bbox.y + region.bbox.h / 2
  const scale = THUMB / side
  return (
    <span
      className="crop"
      role="img"
      aria-label={label}
      style={{
        backgroundImage: `url("${url}")`,
        backgroundSize: `${width * scale}px ${height * scale}px`,
        backgroundPosition: `${-(cx - side / 2) * scale}px ${-(cy - side / 2) * scale}px`,
      }}
    />
  )
}

interface LedgerProps {
  result: PeelResult
  busy: boolean
  flipping: string | null
  onFlip: (region: Region, decision: Decision) => void
  onHighlight: (id: string | null) => void
  highlighted: string | null
}

export function Ledger({ result, busy, flipping, onFlip, onHighlight, highlighted }: LedgerProps) {
  const { ledger } = result
  const original = imageUrl(result.images.original, 2048)
  const edited = imageUrl(result.images.edited, 2048)
  const isGlobal = ledger.mode === 'global'
  const precomputed = result.images.peeled.startsWith('./examples/')

  const downloadLedger = () => {
    const blob = new Blob([JSON.stringify(ledger, null, 2)], { type: 'application/json' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = 'peel-ledger.json'
    link.click()
    URL.revokeObjectURL(link.href)
  }

  return (
    <aside className="ticket" aria-label="Drift ledger" aria-busy={busy}>
      <header className="ticket-head">
        <h2>Drift ledger</h2>
        <p className="ticket-instruction">
          Instruction: <q>{ledger.instruction}</q>
        </p>
        <p className="ticket-headline">{ledger.headline}</p>
      </header>

      <DriftMeter score={ledger.drift_score} />

      <dl className="facts">
        <div>
          <dt>Whole-image colour</dt>
          <dd>{ledger.global_cast.sentence}</dd>
        </div>
        <div>
          <dt>Alignment</dt>
          <dd>{ledger.registration.sentence}</dd>
        </div>
        <div>
          <dt>Untouched pixels</dt>
          <dd>
            {isGlobal ? (
              'Whole-image edit: the frame is meant to change, so nothing is byte-identical.'
            ) : (
              <>
                <Num>{percent(ledger.summary.bit_identical_pct, 1)}</Num> of the result is
                byte-for-byte your original. Check any spot with the loupe.
              </>
            )}
          </dd>
        </div>
      </dl>

      {ledger.warnings.length > 0 && (
        <ul className="warnings">
          {ledger.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      <section className="regions" aria-label="Changed regions">
        <h3>
          <span className="num">{ledger.regions.length}</span> {ledger.regions.length === 1 ? 'region' : 'regions'} changed
        </h3>
        <p className="legend">
          <span>
            <i className="swatch" data-ink="keep" /> Kept
          </span>
          <span>
            <i className="swatch" data-ink="revert" /> Peeled
          </span>
          <span>
            <i className="swatch" data-ink="review" /> Review
          </span>
        </p>
        {ledger.regions.length === 0 && <p className="regions-empty">No region stands out from pixel noise.</p>}
        <ul>
          {ledger.regions.map((region) => {
            const kept = region.decision === 'keep'
            const pending = flipping === region.id
            return (
              <li
                key={region.id}
                className="region"
                data-active={highlighted === region.id}
                onMouseEnter={() => onHighlight(region.id)}
                onMouseLeave={() => onHighlight(null)}
                onFocus={() => onHighlight(region.id)}
                onBlur={() => onHighlight(null)}
              >
                <div className="region-crops">
                  <Crop url={original} region={region} width={result.width} height={result.height} label="This region in the original" />
                  <Crop url={edited} region={region} width={result.width} height={result.height} label="This region in the AI edit" />
                </div>
                <div className="region-body">
                  <p className="region-state">
                    <i className="swatch" data-ink={ink(region)} />
                    <span>{inkWord(region)}</span>
                    <span className="region-type">{TYPE_LABELS[region.type]}</span>
                  </p>
                  <p className="region-sentence">{region.sentence}</p>
                  <p className="region-metrics">
                    <Num>{percent(region.area_pct, 1)}</Num> of frame · ΔE <Num>{fixed(region.mean_delta_e, 1)}</Num> ·
                    LPIPS <Num>{fixed(region.mean_lpips, 2)}</Num>
                    {region.face_distance !== null && region.face_distance !== undefined && (
                      <>
                        {' '}
                        · identity <Num>{fixed(region.face_distance, 2)}</Num>
                      </>
                    )}
                  </p>
                </div>
                <div className="seg region-switch" role="group" aria-label={`Decision for region ${region.id}`}>
                  <button type="button" aria-pressed={kept} disabled={busy || pending} onClick={() => !kept && onFlip(region, 'keep')}>
                    Keep
                  </button>
                  <button type="button" aria-pressed={!kept} disabled={busy || pending} onClick={() => kept && onFlip(region, 'revert')}>
                    Peel
                  </button>
                </div>
              </li>
            )
          })}
        </ul>
      </section>

      <footer className="ticket-foot">
        <div className="ticket-actions">
          <a className="btn btn-solid" href={downloadUrl(result.images.peeled)} download="peeled.png">
            Download peeled image
          </a>
          <button type="button" className="btn" onClick={downloadLedger}>
            Ledger JSON
          </button>
        </div>
        {precomputed ? (
          <p className="timings">This example was analysed ahead of time. Run your own pair to see live timings.</p>
        ) : (
          result.timings.length > 1 && (
            <p className="timings">
              Analysed in <Num>{seconds(result.total_ms)}</Num>
              {result.cached ? ' (served from cache)' : ''}:{' '}
              {result.timings.map((t, i) => (
                <span key={t.stage}>
                  {i > 0 ? ' · ' : ''}
                  {t.stage} <Num>{seconds(t.ms)}</Num>
                </span>
              ))}
            </p>
          )
        )}
        <p className="privacy">
          Images are held in memory for 30 minutes and never written to disk. Faces are compared only within your
          pair; no face data is stored.
        </p>
      </footer>
    </aside>
  )
}
