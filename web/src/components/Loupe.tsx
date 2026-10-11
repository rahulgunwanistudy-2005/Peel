import { useEffect, useRef, useState } from 'react'

export const LOUPE_WINDOW = 33 // source pixels per pane (odd, so there is a centre pixel)
const PANE = 132 // CSS pixels per pane: 4x magnification of the source pixels

export interface LoupeImages {
  original: HTMLImageElement | null
  edited: HTMLImageElement | null
  peeled: HTMLImageElement | null
}

export interface LoupePoint {
  /** Position as a fraction of the image (0..1), so it maps onto images of any decoded size. */
  u: number
  v: number
  /** Position in the viewport, for placing the floating loupe. Null when pinned. */
  clientX: number | null
  clientY: number | null
}

interface Readout {
  peeled: number
  edited: number
  total: number
}

function sourceRect(image: HTMLImageElement, point: LoupePoint) {
  const half = (LOUPE_WINDOW - 1) / 2
  const x = point.u * image.naturalWidth
  const y = point.v * image.naturalHeight
  const sx = Math.min(Math.max(Math.round(x) - half, 0), Math.max(0, image.naturalWidth - LOUPE_WINDOW))
  const sy = Math.min(Math.max(Math.round(y) - half, 0), Math.max(0, image.naturalHeight - LOUPE_WINDOW))
  return { sx, sy }
}

function pixels(image: HTMLImageElement, point: LoupePoint, scratch: HTMLCanvasElement) {
  const ctx = scratch.getContext('2d', { willReadFrequently: true })
  if (!ctx) return null
  const { sx, sy } = sourceRect(image, point)
  ctx.clearRect(0, 0, LOUPE_WINDOW, LOUPE_WINDOW)
  ctx.drawImage(image, sx, sy, LOUPE_WINDOW, LOUPE_WINDOW, 0, 0, LOUPE_WINDOW, LOUPE_WINDOW)
  try {
    return ctx.getImageData(0, 0, LOUPE_WINDOW, LOUPE_WINDOW).data
  } catch {
    return null // canvas tainted: the readout is simply not shown
  }
}

function countDifferent(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  let different = 0
  for (let i = 0; i < a.length; i += 4) {
    if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) different += 1
  }
  return different
}

function Pane({ image, point, label }: { image: HTMLImageElement | null; point: LoupePoint; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    ctx.imageSmoothingEnabled = false
    ctx.fillStyle = '#777777'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    if (!image || !image.complete || image.naturalWidth === 0) return
    const { sx, sy } = sourceRect(image, point)
    ctx.drawImage(image, sx, sy, LOUPE_WINDOW, LOUPE_WINDOW, 0, 0, canvas.width, canvas.height)
  }, [image, point])
  return (
    <figure className="loupe-pane">
      <canvas ref={ref} width={PANE * 2} height={PANE * 2} aria-hidden="true" />
      <figcaption>{label}</figcaption>
    </figure>
  )
}

/**
 * A linen tester: three panes locked to the same spot of the original, the AI edit and the peeled
 * result, at 4x with no smoothing. The readout counts differing pixels in the window, so the
 * "byte-identical outside the edit" claim can be checked by anyone, anywhere on the image.
 */
export function Loupe({ images, point, docked }: { images: LoupeImages; point: LoupePoint; docked: boolean }) {
  const scratch = useRef<HTMLCanvasElement | null>(null)
  const [readout, setReadout] = useState<Readout | null>(null)
  const { original, edited, peeled } = images

  useEffect(() => {
    if (!original || !edited || !peeled) return
    if (!scratch.current) {
      scratch.current = document.createElement('canvas')
      scratch.current.width = LOUPE_WINDOW
      scratch.current.height = LOUPE_WINDOW
    }
    const a = pixels(original, point, scratch.current)
    const o = a ? new Uint8ClampedArray(a) : null
    const e = pixels(edited, point, scratch.current)
    const ed = e ? new Uint8ClampedArray(e) : null
    const p = pixels(peeled, point, scratch.current)
    if (!o || !ed || !p) {
      setReadout(null)
      return
    }
    setReadout({ peeled: countDifferent(o, p), edited: countDifferent(o, ed), total: LOUPE_WINDOW * LOUPE_WINDOW })
  }, [original, edited, peeled, point])

  let style: React.CSSProperties | undefined
  if (!docked && point.clientX !== null && point.clientY !== null) {
    const width = PANE * 3 + 40
    const height = PANE + 96
    const left = point.clientX + 24 + width > window.innerWidth ? point.clientX - 24 - width : point.clientX + 24
    const top = Math.min(Math.max(8, point.clientY - height / 2), window.innerHeight - height - 8)
    style = { left: Math.max(8, left), top }
  }

  return (
    <div className="loupe" data-docked={docked} style={style} role="status" aria-live="off">
      <div className="loupe-panes">
        <Pane image={images.original} point={point} label="Original" />
        <Pane image={images.edited} point={point} label="AI edit" />
        <Pane image={images.peeled} point={point} label="Peeled" />
      </div>
      <p className="loupe-readout num">
        {readout === null ? (
          'Reading pixels'
        ) : (
          <>
            <span data-same={readout.peeled === 0}>
              {readout.peeled === 0
                ? 'Peeled is identical to the original here'
                : `Peeled differs from the original in ${readout.peeled} of ${readout.total} pixels`}
            </span>
            <span>
              {readout.edited === 0
                ? 'The AI edit is identical here too'
                : `The AI edit differs in ${readout.edited} of ${readout.total}`}
            </span>
          </>
        )}
      </p>
      <p className="loupe-where num">
        {Math.round(point.u * 100)}% across, {Math.round(point.v * 100)}% down · 4× · {LOUPE_WINDOW} px window
      </p>
    </div>
  )
}
