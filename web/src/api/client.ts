import bundledExamples from './fixtures/examples.json'
import type { components } from './types'

export type PeelResult = components['schemas']['PeelResult']
export type Ledger = components['schemas']['Ledger']
export type Region = components['schemas']['Region']
export type Example = components['schemas']['Example']
export type Health = components['schemas']['Health']
export type Limits = components['schemas']['Limits']
export type JobCreated = components['schemas']['JobCreated']
export type JobStatus = components['schemas']['JobStatus']
export type ProgressEvent = components['schemas']['ProgressEvent']
export type ChainResult = components['schemas']['ChainResult']
export type ChainStepMetrics = components['schemas']['ChainStepMetrics']
export type ErrorDetail = components['schemas']['ErrorDetail']
export type Decision = 'keep' | 'revert'
export type ImageName = keyof PeelResult['images']

// In development the API runs on its own port. In a production build with no VITE_API_BASE the
// page is being served by the API itself (the Space), so requests are same-origin. The value
// "off" marks a deployment that has no analysis server behind it yet.
const FALLBACK_BASE = import.meta.env.DEV ? 'http://127.0.0.1:7860' : ''
const CONFIGURED: string = import.meta.env.VITE_API_BASE ?? FALLBACK_BASE
export const API_ENABLED = CONFIGURED !== 'off'
export const API_BASE: string = API_ENABLED ? CONFIGURED.replace(/\/$/, '') : ''

/** An error the interface can show as is: the server writes messages for people. */
export class ApiError extends Error {
  code: string
  constructor(code: string, message: string) {
    super(message)
    this.code = code
  }
}

const OFFLINE = new ApiError(
  'OFFLINE',
  'The analysis server is not answering. It sleeps when idle and takes about a minute to wake; try again shortly.',
)

const NOT_CONNECTED = new ApiError(
  'NOT_CONNECTED',
  'This copy of the site has no analysis server connected, so it can only show the bundled examples. Run Peel locally (make dev) to analyse your own images.',
)

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T
  let detail: ErrorDetail | undefined
  try {
    detail = ((await response.json()) as { error?: ErrorDetail }).error
  } catch {
    detail = undefined
  }
  if (!detail) {
    // Not a Peel error body: a proxy, a sleeping host or a static file server answered instead.
    throw response.status === 404 || response.status >= 502 ? OFFLINE : new ApiError('INTERNAL', `The server answered ${response.status}. Try again.`)
  }
  throw new ApiError(detail.code, detail.message)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_ENABLED) throw NOT_CONNECTED
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw OFFLINE
  }
  return unwrap<T>(response)
}

const PHOTO = /\/(original|edited|peeled)\.webp$/

/**
 * API image URLs start with /v1/; bundled example images are relative to the page. Bundled
 * photographs have a lossy display copy and a thumbnail next to the lossless file: pass `maxSide`
 * for display, and nothing when exact pixels are needed (the loupe).
 */
export function imageUrl(url: string, maxSide = 0): string {
  if (!url.startsWith('/v1/')) {
    if (maxSide > 0 && maxSide <= 360 && PHOTO.test(url)) return url.replace(PHOTO, '/thumb.webp')
    if (maxSide > 0 && PHOTO.test(url)) return url.replace(/\.webp$/, '.display.webp')
    return url
  }
  const full = `${API_BASE}${url}`
  if (!maxSide) return full
  return `${full}${full.includes('?') ? '&' : '?'}max=${maxSide}`
}

export function downloadUrl(url: string): string {
  if (!url.startsWith('/v1/')) return url
  const full = `${API_BASE}${url}`
  return `${full}${full.includes('?') ? '&' : '?'}download=true`
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>('/healthz', { signal })
}

export interface JobHandlers {
  onProgress: (event: ProgressEvent) => void
}

/** Follow a job over Server-Sent Events; fall back to polling if the stream breaks. */
function follow(job: JobCreated, handlers: JobHandlers): Promise<JobStatus> {
  return new Promise<JobStatus>((resolve, reject) => {
    let settled = false
    const source = new EventSource(`${API_BASE}${job.events_url}`)
    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      source.close()
      fn()
    }
    source.addEventListener('progress', (e) => {
      handlers.onProgress(JSON.parse((e as MessageEvent<string>).data) as ProgressEvent)
    })
    source.addEventListener('done', (e) => {
      const status = JSON.parse((e as MessageEvent<string>).data) as JobStatus
      finish(() => resolve(status))
    })
    source.addEventListener('error', (e) => {
      const data = (e as MessageEvent<string>).data
      if (typeof data === 'string' && data.length > 0) {
        const detail = JSON.parse(data) as ErrorDetail
        finish(() => reject(new ApiError(detail.code, detail.message)))
        return
      }
      // Transport error: stop the stream and poll the job instead.
      finish(() => {
        poll(job, handlers).then(resolve, reject)
      })
    })
  })
}

async function poll(job: JobCreated, handlers: JobHandlers): Promise<JobStatus> {
  for (;;) {
    const status = await request<JobStatus>(job.status_url)
    handlers.onProgress(status.progress)
    if (status.state === 'done') return status
    if (status.state === 'error') {
      throw new ApiError(status.error?.code ?? 'INTERNAL', status.error?.message ?? 'The job failed.')
    }
    await new Promise((r) => setTimeout(r, 400))
  }
}

export async function peel(
  original: Blob,
  edited: Blob,
  instruction: string,
  handlers: JobHandlers,
): Promise<PeelResult> {
  const form = new FormData()
  form.append('original', original, 'original')
  form.append('edited', edited, 'edited')
  form.append('instruction', instruction)
  const job = await request<JobCreated>('/v1/peel/jobs', { method: 'POST', body: form })
  const status = await follow(job, handlers)
  if (!status.result) throw new ApiError('INTERNAL', 'The job finished without a result.')
  return status.result
}

export async function peelChain(
  original: Blob,
  edits: Blob[],
  instructions: string[],
  handlers: JobHandlers,
): Promise<ChainResult> {
  const form = new FormData()
  form.append('original', original, 'original')
  edits.forEach((edit, i) => form.append('edited', edit, `edit-${i + 1}`))
  form.append('instructions', JSON.stringify(instructions))
  const job = await request<JobCreated>('/v1/chain/jobs', { method: 'POST', body: form })
  const status = await follow(job, handlers)
  if (!status.chain) throw new ApiError('INTERNAL', 'The chain finished without a result.')
  return status.chain
}

export function recomposite(jobId: string, overrides: Record<string, Decision>): Promise<PeelResult> {
  return request<PeelResult>('/v1/recomposite', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId, overrides }),
  })
}

/** Ask the server to edit the image with Nano Banana. Returns the edit and the model that made it. */
export async function editWithModel(original: Blob, instruction: string): Promise<{ image: Blob; model: string }> {
  const form = new FormData()
  form.append('original', original, 'original')
  form.append('instruction', instruction)
  if (!API_ENABLED) throw NOT_CONNECTED
  let response: Response
  try {
    response = await fetch(`${API_BASE}/v1/edit`, { method: 'POST', body: form })
  } catch {
    throw OFFLINE
  }
  if (!response.ok) await unwrap<never>(response)
  return { image: await response.blob(), model: response.headers.get('X-Peel-Edit-Model') ?? 'Gemini' }
}

/**
 * Curated examples are compiled into the bundle (and their images are static files), so the first
 * view needs no request to the server and reserves its layout before anything loads.
 */
export const EXAMPLES = bundledExamples as unknown as Example[]
