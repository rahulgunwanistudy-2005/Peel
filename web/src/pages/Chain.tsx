import { useEffect, useState } from 'react'

import { API_ENABLED, ApiError, ensureHealth, imageUrl, peelChain, type ChainResult, type Limits, type ProgressEvent } from '../api/client'
import { fitImages } from '../lib/fit'
import { LineChart } from '../components/LineChart'
import { Num } from '../components/Num'
import { fixed } from '../lib/format'

interface ChainExample {
  title: string
  source: string
  chain: ChainResult
}

export default function Chain() {
  const [example, setExample] = useState<ChainExample | null>(null)
  const [chain, setChain] = useState<ChainResult | null>(null)
  const [step, setStep] = useState(0)
  const [original, setOriginal] = useState<File | null>(null)
  const [edits, setEdits] = useState<File[]>([])
  const [instructions, setInstructions] = useState('')
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [limits, setLimits] = useState<Limits | null>(null)
  const [stateless, setStateless] = useState(false)

  useEffect(() => {
    if (!API_ENABLED) return
    let live = true
    ensureHealth()
      .then((health) => {
        if (!live) return
        setLimits(health.limits)
        setStateless(health.stateless)
      })
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [])

  useEffect(() => {
    let live = true
    fetch('./chain/chain.json')
      .then((r) => (r.ok ? (r.json() as Promise<ChainExample>) : null))
      .then((data) => {
        if (live && data) {
          setExample(data)
          setChain(data.chain)
          setStep(data.chain.steps.length - 1)
        }
      })
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [])

  const lines = instructions
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
  const ready = original !== null && edits.length > 0 && (lines.length === edits.length || lines.length === 1)

  const run = async () => {
    if (!original || !ready) return
    setProblem(null)
    setProgress({ stage: 'queued', pct: 0, label: 'Sending the images' })
    try {
      const sorted = [...edits].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
      const list = lines.length === 1 ? sorted.map(() => lines[0]) : lines
      const budget = limits && limits.max_request_bytes > 0 ? limits.max_request_bytes - 64_000 : 0
      const fitted = await fitImages([original, ...sorted], budget, 1024)
      const result = await peelChain(fitted.files[0], fitted.files.slice(1), list, { onProgress: setProgress })
      setChain(result)
      setStep(result.steps.length - 1)
    } catch (error) {
      setProblem(error instanceof ApiError ? error.message : 'The chain could not be processed. Try again.')
    } finally {
      setProgress(null)
    }
  }

  const current = chain?.steps[step] ?? null
  const metric = chain?.metrics[step] ?? null
  const isExample = chain !== null && example !== null && chain === example.chain

  return (
    <div className="page">
      <header>
        <h1>Chains</h1>
        <p className="lede">
          Every AI edit re-renders the whole frame, so twenty edits in a row leave you with a background that no longer
          matches the photo you started with. Peel anchors each step to the previous peeled frame: edit 20 still has
          the background of edit 0.
        </p>
      </header>

      {chain && current && metric ? (
        <section className="block" aria-label="Chain result">
          <h2>{isExample && example ? example.title : 'Your chain'}</h2>
          <p>
            Step {current.step} of {chain.steps.length}: <q>{current.instruction}</q>
          </p>
          <div className="film" role="group" aria-label="Steps">
            {chain.steps.map((s, i) => (
              <button key={s.step} type="button" aria-pressed={i === step} onClick={() => setStep(i)}>
                <img src={imageUrl(s.peeled, 320)} alt="" loading="lazy" width={chain.width} height={chain.height} />
                <span>
                  Step {s.step}: {s.instruction}
                </span>
              </button>
            ))}
          </div>
          <div className="pair">
            <figure>
              <img src={imageUrl(current.raw, 1280)} alt={`Raw chain after step ${current.step}`} width={chain.width} height={chain.height} />
              <figcaption>
                <strong>Raw chain</strong>
                <span>
                  Untouched areas vs step 0: LPIPS <Num>{fixed(metric.raw_lpips, 3)}</Num>, PSNR{' '}
                  <Num>{fixed(metric.raw_psnr, 1)}</Num> dB
                </span>
              </figcaption>
            </figure>
            <figure>
              <img src={imageUrl(current.peeled, 1280)} alt={`Peeled chain after step ${current.step}`} width={chain.width} height={chain.height} />
              <figcaption>
                <strong>Peeled chain</strong>
                <span>
                  Untouched areas vs step 0: LPIPS <Num>{fixed(metric.peeled_lpips, 3)}</Num>, PSNR{' '}
                  {metric.peeled_psnr >= 99 ? 'identical' : <Num>{`${fixed(metric.peeled_psnr, 1)} dB`}</Num>}
                </span>
              </figcaption>
            </figure>
          </div>
          <LineChart
            title="Drift in the areas no edit touched, measured against step 0"
            xLabel="Edit step"
            yLabel="LPIPS vs step 0"
            yDigits={3}
            series={[
              { ink: 'raw', label: 'Raw chain', points: chain.metrics.map((m) => ({ x: m.step, y: m.raw_lpips })) },
              { ink: 'peeled', label: 'Peeled chain', points: chain.metrics.map((m) => ({ x: m.step, y: m.peeled_lpips })) },
            ]}
          />
          {isExample && example && <p>Source: {example.source}</p>}
        </section>
      ) : (
        <p className="empty-data">No chain loaded yet. Upload one below.</p>
      )}

      <section className="block" aria-label="Run your own chain">
        <h2>Run your own chain</h2>
        <p>
          Upload the step-0 image and the editor's outputs in order (file names are sorted, so name them 01, 02, 03).
          Up to 8 steps.
        </p>
        {(stateless || !API_ENABLED) && (
          <p className="empty-data">
            Chains need a server that keeps each step in memory, and the public demo server does not. The example
            above is real; to run your own chain, start Peel locally with <code>make dev</code>.
          </p>
        )}
        <form
          className="chain-form"
          onSubmit={(e) => {
            e.preventDefault()
            void run()
          }}
        >
          <label>
            Step-0 image
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => setOriginal(e.target.files?.[0] ?? null)} />
          </label>
          <label>
            Edited images, in order
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              multiple
              onChange={(e) => setEdits(Array.from(e.target.files ?? []).slice(0, 8))}
            />
          </label>
          <label>
            Instructions
            <span>One per line, in the same order. A single line is used for every step.</span>
            <textarea value={instructions} onChange={(e) => setInstructions(e.target.value)} placeholder={'Add an apple on the table\nAdd a banana on the table'} />
          </label>
          {problem && (
            <p className="upload-problem" role="alert">
              {problem}
            </p>
          )}
          <div className="upload-actions">
            <button type="submit" className="btn btn-solid" disabled={!ready || progress !== null || stateless || !API_ENABLED}>
              {progress ? progress.label || 'Working' : 'Peel the chain'}
            </button>
            <p className="upload-need" aria-live="polite">
              {ready
                ? `${edits.length} ${edits.length === 1 ? 'step' : 'steps'} ready.`
                : `Needed: a step-0 image, at least one edit, and ${edits.length > 1 ? `${edits.length} instructions (or one for all)` : 'an instruction'}.`}
            </p>
          </div>
        </form>
      </section>
    </div>
  )
}
