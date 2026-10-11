import { useCallback, useEffect, useRef, useState } from 'react'

import {
  ApiError,
  editWithModel,
  getHealth,
  loadExamples,
  peel,
  recomposite,
  type Decision,
  type Example,
  type PeelResult,
  type ProgressEvent,
  type Region,
} from '../api/client'
import { ExampleStrip } from '../components/ExampleStrip'
import { Ledger } from '../components/Ledger'
import { Progress } from '../components/Progress'
import { Stage, type Overlay, type View } from '../components/Stage'
import { UploadPanel } from '../components/UploadPanel'

interface Notice {
  code: string
  message: string
}

function toNotice(error: unknown): Notice {
  if (error instanceof ApiError) return { code: error.code, message: error.message }
  return { code: 'INTERNAL', message: 'Something unexpected happened in the page. Reload and try again.' }
}

export function Workbench() {
  const [examples, setExamples] = useState<Example[]>([])
  const [example, setExample] = useState<Example | null>(null)
  const [result, setResult] = useState<PeelResult | null>(null)
  const [playKey, setPlayKey] = useState('initial')
  const [view, setView] = useState<View>('peeled')
  const [overlay, setOverlay] = useState<Overlay>('map')
  const [highlighted, setHighlighted] = useState<string | null>(null)
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [flipping, setFlipping] = useState<string | null>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [own, setOwn] = useState(false)
  const [editModel, setEditModel] = useState<string | null>(null)
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'empty'>('loading')
  const noticeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let live = true
    loadExamples()
      .then((list) => {
        if (!live) return
        setExamples(list)
        if (list.length > 0) {
          setExample(list[0])
          setResult(list[0].result)
          setPlayKey(`example-${list[0].id}`)
          setLoadState('ready')
        } else {
          setLoadState('empty')
          setOwn(true)
        }
      })
      .catch(() => {
        if (live) {
          setLoadState('empty')
          setOwn(true)
        }
      })
    return () => {
      live = false
    }
  }, [])

  // Only to learn whether live editing is offered. The page works without the server.
  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then((health) => setEditModel(health.edit.available ? (health.edit.model ?? 'Gemini') : null))
      .catch(() => setEditModel(null))
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (notice) noticeRef.current?.focus()
  }, [notice])

  const show = useCallback((next: PeelResult, key: string) => {
    setResult(next)
    setPlayKey(key)
    setView('peeled')
    setOverlay('map')
    setHighlighted(null)
  }, [])

  const selectExample = (next: Example) => {
    setNotice(null)
    setExample(next)
    show(next.result, `example-${next.id}-${Date.now()}`)
  }

  const runPeel = async (original: Blob, edited: Blob, instruction: string) => {
    setNotice(null)
    setProgress({ stage: 'queued', pct: 0, label: 'Sending the images' })
    try {
      const next = await peel(original, edited, instruction, { onProgress: setProgress })
      setExample(null)
      show(next, `job-${next.id}`)
    } catch (error) {
      setNotice(toNotice(error))
    } finally {
      setProgress(null)
    }
  }

  const runEdit = async (original: Blob, instruction: string): Promise<Blob | null> => {
    setNotice(null)
    try {
      const { image } = await editWithModel(original, instruction)
      return image
    } catch (error) {
      setNotice(toNotice(error))
      return null
    }
  }

  const flip = async (region: Region, decision: Decision) => {
    if (!result) return
    setNotice(null)
    setFlipping(region.id)
    const overrides: Record<string, Decision> = {}
    for (const r of result.ledger.regions) {
      if (r.overridden && r.id !== region.id) overrides[r.id] = r.decision
    }
    const wasOverridden = region.overridden
    if (!wasOverridden) overrides[region.id] = decision // flipping back restores the classifier's choice
    try {
      const next = await recomposite(result.id, overrides)
      setResult(next)
    } catch (error) {
      setNotice(toNotice(error))
    } finally {
      setFlipping(null)
    }
  }

  const busy = progress !== null

  return (
    <div className="workbench">
      <h1 className="visually-hidden">Peel workbench</h1>

      {notice && (
        <div className="notice" role="alert" tabIndex={-1} ref={noticeRef}>
          <p>{notice.message}</p>
          <button type="button" className="btn btn-small" onClick={() => setNotice(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="bench">
        <div className="bench-main">
          {result ? (
            <Stage
              result={result}
              view={view}
              overlay={overlay}
              onView={setView}
              onOverlay={setOverlay}
              highlighted={highlighted}
              playKey={playKey}
              busyStage={progress?.stage ?? null}
            />
          ) : (
            <section className="stage stage-empty" aria-label="Image stage">
              <div className="stage-well">
                <p>
                  {loadState === 'loading'
                    ? 'Loading the example'
                    : 'Add an original, its AI edit and the instruction below. The result appears here.'}
                </p>
              </div>
            </section>
          )}

          {progress && <Progress event={progress} />}

          {own ? (
            <UploadPanel
              busy={busy}
              editModel={editModel}
              onPeel={runPeel}
              onEdit={runEdit}
              onClose={() => setOwn(false)}
            />
          ) : (
            <ExampleStrip
              examples={examples}
              selected={example?.id ?? null}
              disabled={busy}
              onSelect={selectExample}
              onOwn={() => setOwn(true)}
            />
          )}

          {example && !own && (
            <p className="example-source">
              {example.blurb} Source: {example.source}
            </p>
          )}
        </div>

        {result && (
          <Ledger
            result={result}
            busy={busy}
            flipping={flipping}
            onFlip={flip}
            onHighlight={setHighlighted}
            highlighted={highlighted}
          />
        )}
      </div>
    </div>
  )
}
