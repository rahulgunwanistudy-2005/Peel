import { API_BASE as CONFIGURED_BASE, API_ENABLED } from '../api/client'
import { GITHUB_URL } from '../components/TopBar'

// What to print in the snippets: the connected API, this origin when the API serves the page
// itself, or the local dev address when this copy of the site has no API behind it.
const API_BASE = !API_ENABLED ? 'http://127.0.0.1:7860' : CONFIGURED_BASE || window.location.origin

const CURL = `curl -s "${API_BASE}/v1/peel?inline=base64" \\
  -F original=@original.png \\
  -F edited=@edited.png \\
  -F 'instruction=Make the jacket red' > result.json

jq '.ledger.headline, .ledger.drift_score' result.json
jq -r .images.peeled result.json | base64 --decode > peeled.webp`

const PYTHON = `import base64
import requests

API = "${API_BASE}"

with open("original.png", "rb") as original, open("edited.png", "rb") as edited:
    response = requests.post(
        f"{API}/v1/peel",
        params={"inline": "base64"},  # images come back inside the response (WebP)
        files={"original": original, "edited": edited},
        data={"instruction": "Make the jacket red"},
        timeout=180,
    )
result = response.json()
if "error" in result:
    raise SystemExit(result["error"]["message"])

ledger = result["ledger"]
print(ledger["headline"], "| drift score", ledger["drift_score"])
for region in ledger["regions"]:
    print(region["decision"], "-", region["sentence"])

open("peeled.webp", "wb").write(base64.b64decode(result["images"]["peeled"]))

# Keep a region the classifier peeled: send the pair again with the decision.
# (On a server that keeps results, POST /v1/recomposite does this without re-analysing.)`

const ENDPOINTS: [string, string][] = [
  ['POST /v1/peel', 'Original, edit and instruction in; ledger, timings and images out. Add ?inline=base64 to get the images in the response.'],
  ['POST /v1/peel/stream', 'The same in one streamed response: stage progress, then the result. This page uses it.'],
  ['POST /v1/peel/jobs', 'Start a job and follow GET /v1/jobs/{id}/events. Needs a server that keeps results in memory.'],
  ['POST /v1/recomposite', 'Flip regions between keep and revert on a finished job. No re-analysis.'],
  ['POST /v1/chain', 'A sequence of chained edits, anchored to the step-0 image, with a drift curve.'],
  ['POST /v1/edit', 'Edit an image with Nano Banana on the server. Answers EDIT_UNAVAILABLE when no key is set.'],
  ['GET /v1/examples', 'Curated pairs with precomputed results.'],
  ['GET /healthz', 'Status, model warm flags, size limits, and whether the host is stateless.'],
]

export default function ApiPage() {
  return (
    <div className="page">
      <header>
        <h1>API</h1>
        <p className="lede">
          Everything the workbench does is one HTTP call. Images are kept in memory for 30 minutes and never written
          to disk. Errors come back as a code and a sentence you can show to a person.
        </p>
        {!API_ENABLED && (
          <p className="empty-data">
            This copy of the site has no analysis server connected. The snippets below use the local address you get
            from <code>make dev</code>.
          </p>
        )}
      </header>

      <section className="block">
        <h2>Endpoints</h2>
        <ul className="endpoint-list">
          {ENDPOINTS.map(([name, text]) => (
            <li key={name}>
              <code>{name}</code>
              <span>{text}</span>
            </li>
          ))}
        </ul>
        <div className="link-row">
          <a className="btn btn-solid" href={`${API_BASE}/docs`} rel="noreferrer">
            Interactive docs
          </a>
          <a className="btn" href={`${API_BASE}/openapi.json`} rel="noreferrer">
            OpenAPI schema
          </a>
          <a className="btn" href={GITHUB_URL} rel="noreferrer">
            Source on GitHub
          </a>
        </div>
      </section>

      <section className="block">
        <h2>curl</h2>
        <pre className="code" tabIndex={0}>
          <code>{CURL}</code>
        </pre>
      </section>

      <section className="block">
        <h2>Python</h2>
        <pre className="code" tabIndex={0}>
          <code>{PYTHON}</code>
        </pre>
      </section>

      <section className="block">
        <h2>n8n: a drift gate for automated edits</h2>
        <p>
          Workflows that chain Nano Banana edits on product photos promise that the product stays unchanged. This
          workflow checks. It sends every AI edit through Peel, passes it when the drift score is under your
          threshold, and otherwise swaps in the repaired image and posts the ledger summary to your team.
        </p>
        <div className="link-row">
          <a className="btn btn-solid" href="./peel-gate.workflow.json" download>
            Download the n8n workflow
          </a>
          <a className="btn" href={`${GITHUB_URL}/tree/main/integrations/n8n`} rel="noreferrer">
            Setup notes
          </a>
        </div>
      </section>
    </div>
  )
}
