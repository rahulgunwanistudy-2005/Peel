import { LineChart } from '../components/LineChart'

/**
 * Every number on this page is read from eval/results/*.json, copied into src/data/eval by
 * `make eval`. Nothing here is typed in by hand; a missing file renders as "not measured yet".
 */
const FILES = import.meta.glob<unknown>('../data/eval/*.json', { eager: true, import: 'default' })

function load<T>(name: string): T | null {
  return (FILES[`../data/eval/${name}.json`] as T | undefined) ?? null
}

interface MetricInfo {
  key: string
  label: string
  better: 'higher' | 'lower'
  digits: number
  unit: string
}

interface Variant {
  id: string
  label: string
  note: string
  metrics: Record<string, number>
}

interface Synthetic {
  meta: { generated_at: string; git_sha: string; n_cases: number; bases: string[]; drift_kinds: string[] }
  metric_info: MetricInfo[]
  variants: Variant[]
}

interface ChainPoint {
  step: number
  raw_lpips: number
  peeled_lpips: number
  raw_psnr: number
  peeled_psnr: number
}

interface ChainGroup {
  id: string
  label: string
  note: string
  chains: { image: string; chain: string; run: number; steps: ChainPoint[] }[]
  mean_curve: (ChainPoint & { n_chains: number })[]
  checkpoints: (ChainPoint & { n_chains: number })[]
}

interface Banana {
  meta: { generated_at: string; dataset: string; license: string; note: string }
  groups: ChainGroup[]
}

interface RealPairs {
  meta: { n: number; generated_at?: string }
  skipped?: string
  summary?: { intended_recall: number; collateral_revert_precision: number; mean_drift_score: number }
}

interface Latency {
  meta: { hardware: string; n: number; image: string; generated_at: string }
  stages: { stage: string; p50: number; p95: number }[]
  total: { p50: number; p95: number }
}

function Missing({ what }: { what: string }) {
  return <p className="empty-data">Not measured yet: {what}. Run <code>make eval</code> to produce it.</p>
}

function best(variants: Variant[], info: MetricInfo): number {
  const values = variants.filter((v) => v.id !== 'raw').map((v) => v.metrics[info.key])
  return info.better === 'higher' ? Math.max(...values) : Math.min(...values)
}

function SyntheticTable({ data, ids, caption }: { data: Synthetic; ids: string[]; caption: string }) {
  const rows = ids.map((id) => data.variants.find((v) => v.id === id)).filter((v): v is Variant => Boolean(v))
  return (
    <div className="table-wrap">
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Method</th>
            {data.metric_info.map((info) => (
              <th key={info.key} scope="col">
                {info.label}
                {info.unit ? ` (${info.unit})` : ''} {info.better === 'higher' ? '↑' : '↓'}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((variant) => (
            <tr key={variant.id} data-us={variant.id === 'peel'}>
              <td>
                {variant.label}
                <br />
                <small>{variant.note}</small>
              </td>
              {data.metric_info.map((info) => {
                const value = variant.metrics[info.key]
                return (
                  <td key={info.key} className="num" data-best={variant.id !== 'raw' && value === best(rows, info)}>
                    {value.toFixed(info.digits)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Numbers() {
  const synthetic = load<Synthetic>('synthetic')
  const banana = load<Banana>('banana100')
  const real = load<RealPairs>('real_pairs')
  const latency = load<Latency>('latency') ?? load<Latency>('latency_local')

  return (
    <div className="page">
      <header>
        <h1>Numbers</h1>
        <p className="lede">
          Measured by code in the repository and reproducible with one command. Each table says what was measured
          and what it cannot tell you.
        </p>
      </header>

      <section className="block">
        <h2>Synthetic drift suite</h2>
        {synthetic ? (
          <>
            <p>
              <span className="num">{synthetic.meta.n_cases}</span> cases with ground truth by construction. Each
              takes a photo, applies one requested edit inside a known mask (add, remove, recolour or replace an
              object) and then unrequested drift everywhere else ({synthetic.meta.drift_kinds.join(', ')}). The
              ideal answer is known exactly: the photo with only the requested edit.
            </p>
            <SyntheticTable
              data={synthetic}
              ids={['raw', 'diff_merge', 'peel']}
              caption="Raw edit, the diff-merge baseline (composite everything whose pixel difference exceeds a threshold, as existing ComfyUI nodes do) and Peel. The best repair in each column is underlined."
            />
          </>
        ) : (
          <Missing what="the synthetic suite" />
        )}
      </section>

      <section className="block">
        <h2>Ablations</h2>
        {synthetic ? (
          <>
            <p>
              The same suite with one part of Peel switched off at a time. This is the argument for each part: if
              removing it does not hurt a number, it is not earning its place.
            </p>
            <SyntheticTable
              data={synthetic}
              ids={['peel', 'no_rereg', 'no_intent', 'no_cast', 'no_face', 'diff_merge']}
              caption="Full Peel against Peel without background re-registration, without instruction grounding, without cast separation and without the face lock."
            />
          </>
        ) : (
          <Missing what="the ablation table" />
        )}
      </section>

      <section className="block">
        <h2>Banana100 chains</h2>
        {banana && banana.groups.length > 0 ? (
          <>
            <p>
              Real Nano Banana Pro edit chains from the {banana.meta.dataset} dataset ({banana.meta.license}). Each
              chain is measured against its step-0 image, in the areas no edit touched. {banana.meta.note}
            </p>
            {banana.groups.map((group) => (
              <div key={group.id} className="subblock">
                <h3>
                  {group.label} <span className="num">(n = {group.chains.length})</span>
                </h3>
                <p>{group.note}</p>
                <LineChart
                  title={`${group.label}: drift of untouched areas against step 0`}
                  xLabel="Edit step"
                  yLabel="LPIPS vs step 0"
                  yDigits={3}
                  series={[
                    { ink: 'raw', label: 'Raw chain', points: group.mean_curve.map((s) => ({ x: s.step, y: s.raw_lpips })) },
                    { ink: 'peeled', label: 'Peeled chain', points: group.mean_curve.map((s) => ({ x: s.step, y: s.peeled_lpips })) },
                  ]}
                />
                <div className="table-wrap">
                  <table>
                    <caption>Untouched-region quality against the step-0 image at selected steps.</caption>
                    <thead>
                      <tr>
                        <th scope="col">Step</th>
                        <th scope="col">Raw LPIPS ↓</th>
                        <th scope="col">Peeled LPIPS ↓</th>
                        <th scope="col">Raw PSNR (dB) ↑</th>
                        <th scope="col">Peeled PSNR (dB) ↑</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.checkpoints.map((c) => (
                        <tr key={c.step}>
                          <td className="num">{c.step}</td>
                          <td className="num">{c.raw_lpips.toFixed(3)}</td>
                          <td className="num">{c.peeled_lpips.toFixed(3)}</td>
                          <td className="num">{c.raw_psnr.toFixed(1)}</td>
                          <td className="num">{c.peeled_psnr >= 99 ? 'identical' : c.peeled_psnr.toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </>
        ) : (
          <Missing what="the Banana100 chain run" />
        )}
      </section>

      <section className="block">
        <h2>Real pairs</h2>
        {real && real.summary ? (
          <>
            <p>
              <span className="num">{real.meta.n}</span> Nano Banana edits made by hand, each with a rough mask of
              the region that was meant to change.
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Measure</th>
                    <th scope="col">Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Intended recall: share of the hand-drawn region that Peel kept</td>
                    <td className="num">{real.summary.intended_recall.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td>Collateral-revert precision: share of what Peel restored that lies outside the hand-drawn region</td>
                    <td className="num">{real.summary.collateral_revert_precision.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td>Mean drift score of the raw edits</td>
                    <td className="num">{real.summary.mean_drift_score.toFixed(1)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="empty-data">
            {real?.skipped ?? 'Not measured yet: no hand-labelled pairs are in the repository.'} No number is shown
            rather than an invented one.
          </p>
        )}
      </section>

      <section className="block">
        <h2>Latency</h2>
        {latency ? (
          <>
            <p>
              <span className="num">{latency.meta.n}</span> runs on {latency.meta.image}, measured on{' '}
              {latency.meta.hardware}. No GPU.
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Stage</th>
                    <th scope="col">p50 (ms)</th>
                    <th scope="col">p95 (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {latency.stages.map((s) => (
                    <tr key={s.stage}>
                      <td>{s.stage}</td>
                      <td className="num">{s.p50.toFixed(0)}</td>
                      <td className="num">{s.p95.toFixed(0)}</td>
                    </tr>
                  ))}
                  <tr data-us="true">
                    <td>End to end</td>
                    <td className="num">{latency.total.p50.toFixed(0)}</td>
                    <td className="num">{latency.total.p95.toFixed(0)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <Missing what="latency" />
        )}
      </section>
    </div>
  )
}
