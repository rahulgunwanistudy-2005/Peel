import { API_BASE } from '../api/client'
import { GITHUB_URL } from '../components/TopBar'

const CURL = `curl -s ${API_BASE}/v1/peel \\
  -F original=@original.png \\
  -F edited=@edited.png \\
  -F 'instruction=Make the jacket red' \\
  | tee result.json | jq '.ledger.headline, .ledger.drift_score'

# The result lists image URLs relative to the API root:
curl -s -o peeled.png "${API_BASE}$(jq -r .images.peeled result.json)"`

const PYTHON = `import requests

API = "${API_BASE}"

with open("original.png", "rb") as original, open("edited.png", "rb") as edited:
    response = requests.post(
        f"{API}/v1/peel",
        files={"original": original, "edited": edited},
        data={"instruction": "Make the jacket red"},
        timeout=120,
    )
result = response.json()
if "error" in result:
    raise SystemExit(result["error"]["message"])

ledger = result["ledger"]
print(ledger["headline"], "| drift score", ledger["drift_score"])
for region in ledger["regions"]:
    print(region["decision"], "-", region["sentence"])

open("peeled.png", "wb").write(requests.get(API + result["images"]["peeled"]).content)

# Flip a region and recomposite without re-analysing:
flipped = requests.post(
    f"{API}/v1/recomposite",
    json={"job_id": result["id"], "overrides": {"r2": "keep"}},
).json()`

const ENDPOINTS: [string, string][] = [
  ['POST /v1/peel', 'Original, edit and instruction in; ledger, timings and image URLs out.'],
  ['POST /v1/peel/jobs', 'Same, but returns at once. Follow GET /v1/jobs/{id}/events (Server-Sent Events) for stage progress.'],
  ['POST /v1/recomposite', 'Flip regions between keep and revert on a finished job. No re-analysis.'],
  ['POST /v1/chain', 'A sequence of chained edits, anchored to the step-0 image, with a drift curve.'],
  ['POST /v1/edit', 'Edit an image with Nano Banana on the server. Answers EDIT_UNAVAILABLE when no key is set.'],
  ['GET /v1/examples', 'Curated pairs with precomputed results.'],
  ['GET /healthz', 'Status, model warm flags, version.'],
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
