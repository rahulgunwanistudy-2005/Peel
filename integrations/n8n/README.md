# Peel gate for n8n

Many published n8n templates chain Nano Banana edits on product photos and promise that the product
itself stays unchanged. Nothing in those workflows checks. This one does.

```
Form (photo + instruction)
  → Gemini image edit (HTTP Request, generateContent)
  → Peel /v1/peel (multipart: original, edited, instruction)
  → drift score ≤ limit ?
        yes → return the AI edit
        no  → return Peel's repaired image, post the ledger summary to chat
```

## Import

1. In n8n: **Workflows → Import from file** and choose [`peel-gate.workflow.json`](peel-gate.workflow.json).
2. Open **Config** and set:
   - `peelApi`: the Peel API URL, without a trailing slash;
   - `geminiModel`: an image-editing model your key can use (for example `gemini-2.5-flash-image`);
   - `maxDriftScore`: the gate. 5 is strict; raise it if harmless re-rendering trips it;
   - `chatWebhook` (optional): a Slack or Discord incoming-webhook URL.
3. On **Edit with Nano Banana**, select your *Google Gemini(PaLM) Api* credential.
4. Activate the workflow and open the form URL shown on **On form submission**.

## What Peel returns to the workflow

`ledger.drift_score` (0–100), `ledger.headline`, one sentence per changed region, and image URLs
relative to the API root (`images.peeled`, `images.edited`, `images.change_map`). The workflow
fetches the image it decides to keep with a second HTTP Request.

## Notes

- The Peel API allows browser calls from `*.n8n.cloud`; server-side calls from n8n need no CORS.
- Peel keeps images in memory for 30 minutes. Fetch the image in the same run, as this workflow does.
- No Gemini node is required: the edit is a plain HTTP Request to the `generateContent` REST
  endpoint, so the workflow works on any n8n version with the HTTP Request node (v4.2).

## Troubleshooting

- *Photo to base64* says there is no binary data: open **Config → Options** and make sure binary
  data is kept ("Strip Binary Data" off, or "Include Binary File" on, depending on your n8n version).
- *Peel: measure the drift* answers `NOT_SAME_SCENE`: the model returned a different picture, not an
  edit of yours. That is a failed edit; rerun it.
- `RATE_LIMITED`: the public demo API allows a burst of 12 requests per address, then one every
  five seconds.

## Status

The workflow file is checked in as valid n8n export JSON. It has to be imported and run once on
n8n Cloud by someone with an account; screenshots of that run go in `screenshots/`.
