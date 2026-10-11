import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { imageUrl, type PeelResult, type Region } from '../api/client'
import { Loupe, type LoupeImages, type LoupePoint } from './Loupe'
import { RegistrationMarks, type MarkPhase } from './RegistrationMarks'

export type View = 'original' | 'edited' | 'peeled' | 'compare'
export type Overlay = 'none' | 'map' | 'diff'

const DISPLAY_MAX = 2048

const VIEW_LABELS: Record<View, string> = {
  original: 'Original',
  edited: 'AI edit',
  peeled: 'Peeled',
  compare: 'Compare',
}

const OVERLAY_LABELS: Record<Overlay, string> = {
  none: 'Off',
  map: 'Change map',
  diff: 'Difference',
}

function useImage(url: string | null): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  useEffect(() => {
    setImage(null)
    if (!url) return
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.decoding = 'async'
    let live = true
    img.onload = () => {
      if (live) setImage(img)
    }
    img.src = url
    return () => {
      live = false
    }
  }, [url])
  return image
}

function inkFor(region: Region): 'keep' | 'revert' | 'review' {
  if (region.classification === 'ambiguous' && !region.overridden) return 'review'
  return region.decision === 'keep' ? 'keep' : 'revert'
}

interface StageProps {
  result: PeelResult
  view: View
  overlay: Overlay
  onView: (view: View) => void
  onOverlay: (overlay: Overlay) => void
  highlighted: string | null
  /** Changes whenever a new analysis is shown; replays the register-then-reveal moment. */
  playKey: string
  /** Set while a job is running: holds the marks out of register until alignment is done. */
  busyStage: string | null
}

export function Stage({ result, view, overlay, onView, onOverlay, highlighted, playKey, busyStage }: StageProps) {
  const frame = useRef<HTMLDivElement>(null)
  const [split, setSplit] = useState(50)
  const [point, setPoint] = useState<LoupePoint | null>(null)
  const [pinned, setPinned] = useState(false)
  const [loupeWanted, setLoupeWanted] = useState(false)
  const [phase, setPhase] = useState<MarkPhase>('out')
  const [reveal, setReveal] = useState(false)
  const [loaded, setLoaded] = useState(false)

  const urls = useMemo(
    () => ({
      original: imageUrl(result.images.original, DISPLAY_MAX),
      edited: imageUrl(result.images.edited, DISPLAY_MAX),
      peeled: imageUrl(result.images.peeled, DISPLAY_MAX),
      map: imageUrl(result.images.change_map),
      diff: imageUrl(result.images.diff),
      rawDiff: imageUrl(result.images.raw_diff),
    }),
    [result.images],
  )

  // The loupe needs pixel access to all three images; fetch them once someone reaches for it.
  const loupeImages: LoupeImages = {
    original: useImage(loupeWanted ? urls.original : null),
    edited: useImage(loupeWanted ? urls.edited : null),
    peeled: useImage(loupeWanted ? urls.peeled : null),
  }

  // Register, then reveal. Runs on first load and for every new analysis.
  useEffect(() => {
    setLoaded(false)
    setReveal(false)
    setPhase('out')
  }, [playKey])

  useEffect(() => {
    if (busyStage !== null) {
      setPhase(busyStage === 'queued' || busyStage === 'decode' ? 'out' : busyStage === 'register' ? 'registering' : 'registered')
      return
    }
    if (!loaded) return
    setPhase('registering')
    const toRegistered = window.setTimeout(() => setPhase('registered'), 700)
    const toReveal = window.setTimeout(() => setReveal(true), 760)
    return () => {
      window.clearTimeout(toRegistered)
      window.clearTimeout(toReveal)
    }
  }, [busyStage, loaded, playKey])

  const baseUrl = view === 'original' ? urls.original : view === 'edited' ? urls.edited : urls.peeled
  const overlayUrl =
    overlay === 'map'
      ? urls.map
      : overlay === 'diff'
        ? view === 'edited'
          ? urls.rawDiff
          : view === 'original'
            ? null
            : urls.diff
        : null

  const locate = useCallback(
    (clientX: number, clientY: number): LoupePoint | null => {
      const rect = frame.current?.getBoundingClientRect()
      if (!rect || rect.width === 0) return null
      const u = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1)
      const v = Math.min(Math.max((clientY - rect.top) / rect.height, 0), 1)
      return { u, v, clientX, clientY }
    },
    [],
  )

  const onPointerMove = (event: React.PointerEvent) => {
    if (event.pointerType !== 'mouse' || pinned) return
    setLoupeWanted(true)
    setPoint(locate(event.clientX, event.clientY))
  }

  const onPointerLeave = (event: React.PointerEvent) => {
    if (event.pointerType === 'mouse' && !pinned) setPoint(null)
  }

  const onClick = (event: React.MouseEvent) => {
    if ((event.target as HTMLElement).closest('.compare-range')) return
    const located = locate(event.clientX, event.clientY)
    if (!located) return
    setLoupeWanted(true)
    // A click or tap pins the loupe to that spot (docked); clicking again releases it.
    if (pinned) {
      setPinned(false)
      setPoint(null)
    } else {
      setPinned(true)
      setPoint({ ...located, clientX: null, clientY: null })
    }
  }

  const onKeyDown = (event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 1 : 12
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    }
    if (event.key === 'Escape') {
      setPinned(false)
      setPoint(null)
      return
    }
    const move = moves[event.key]
    if (!move) return
    event.preventDefault()
    setLoupeWanted(true)
    setPinned(true)
    setPoint((current) => {
      const base = current ?? { u: 0.5, v: 0.5, clientX: null, clientY: null }
      return {
        u: Math.min(Math.max(0, base.u + move[0] / result.width), 1),
        v: Math.min(Math.max(0, base.v + move[1] / result.height), 1),
        clientX: null,
        clientY: null,
      }
    })
  }

  const highlightedRegion = result.ledger.regions.find((r) => r.id === highlighted) ?? null
  const ratio = `${result.width} / ${result.height}`
  const showOverlay = overlayUrl !== null && (overlay !== 'map' || reveal)
  const alt = `${VIEW_LABELS[view === 'compare' ? 'peeled' : view]} image for the instruction: ${result.ledger.instruction}`

  return (
    <section className="stage" aria-label="Image stage">
      <div className="stage-well">
        <div
          className="stage-frame"
          ref={frame}
          style={{ aspectRatio: ratio, width: `min(100%, calc(var(--stage-max-h) * ${result.width / result.height}))` }}
          tabIndex={0}
          role="group"
          aria-label="Image. Hover, tap, or use the arrow keys to inspect pixels with the loupe; Escape closes it."
          onPointerMove={onPointerMove}
          onPointerLeave={onPointerLeave}
          onClick={onClick}
          onKeyDown={onKeyDown}
        >
          <img
            key={playKey /* remount per analysis so onLoad fires even when the URL is unchanged */}
            className="layer"
            src={view === 'compare' ? urls.edited : baseUrl}
            alt={view === 'compare' ? `AI edit for the instruction: ${result.ledger.instruction}` : alt}
            width={result.width}
            height={result.height}
            crossOrigin="anonymous"
            decoding="async"
            fetchPriority="high"
            onLoad={() => setLoaded(true)}
            draggable={false}
          />
          {view === 'compare' && (
            <img
              className="layer"
              src={urls.peeled}
              alt={alt}
              width={result.width}
              height={result.height}
              crossOrigin="anonymous"
              style={{ clipPath: `inset(0 0 0 ${split}%)` }}
              draggable={false}
            />
          )}
          {view === 'compare' && overlay === 'diff' && (
            <img
              className="layer overlay"
              data-kind="diff"
              src={urls.rawDiff}
              alt="Amplified difference between the AI edit and the original"
              crossOrigin="anonymous"
              style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
              draggable={false}
            />
          )}
          {showOverlay && (
            <img
              key={`${overlay}-${overlayUrl}`}
              className="layer overlay"
              data-kind={overlay}
              src={overlayUrl}
              alt={
                overlay === 'map'
                  ? 'Change map: kept regions in cyan, peeled regions in magenta, regions to review in yellow'
                  : 'Amplified difference from the original; black means identical'
              }
              crossOrigin="anonymous"
              style={view === 'compare' ? { clipPath: `inset(0 0 0 ${split}%)` } : undefined}
              draggable={false}
            />
          )}
          {highlightedRegion && (
            <svg className="layer outline" viewBox={`0 0 ${result.width} ${result.height}`} preserveAspectRatio="none" aria-hidden="true">
              <polygon
                points={highlightedRegion.polygon.map(([x, y]) => `${x},${y}`).join(' ')}
                data-ink={inkFor(highlightedRegion)}
              />
            </svg>
          )}
          {view === 'compare' && (
            <>
              <div className="compare-line" style={{ left: `${split}%` }} aria-hidden="true">
                <span>AI edit</span>
                <span>Peeled</span>
              </div>
              <input
                className="compare-range"
                type="range"
                min={0}
                max={100}
                step={0.5}
                value={split}
                aria-label="Position of the divider between the AI edit (left) and the peeled result (right)"
                onChange={(e) => setSplit(Number(e.target.value))}
              />
            </>
          )}
          <RegistrationMarks phase={phase} />
        </div>
      </div>

      <div className="slug">
        <div className="slug-group">
          <span className="slug-label" id="view-label">
            View
          </span>
          <div className="seg" role="group" aria-labelledby="view-label">
            {(Object.keys(VIEW_LABELS) as View[]).map((v) => (
              <button key={v} type="button" aria-pressed={view === v} onClick={() => onView(v)}>
                {VIEW_LABELS[v]}
              </button>
            ))}
          </div>
        </div>
        <div className="slug-group">
          <span className="slug-label" id="overlay-label">
            Overlay
          </span>
          <div className="seg" role="group" aria-labelledby="overlay-label">
            {(Object.keys(OVERLAY_LABELS) as Overlay[]).map((o) => (
              <button key={o} type="button" aria-pressed={overlay === o} onClick={() => onOverlay(o)}>
                {OVERLAY_LABELS[o]}
              </button>
            ))}
          </div>
        </div>
        <p className="slug-note num">
          {result.width} × {result.height} px ·{' '}
          {pinned ? 'Loupe pinned. Click the image or press Escape to release.' : 'Hover or tap the image for the loupe.'}
        </p>
      </div>

      {point && <Loupe images={loupeImages} point={point} docked={pinned} />}
    </section>
  )
}
