# Video script

Target 2:40, hard limit 3:00. Screen recording of the live site at 1440 × 900 with voice-over.
Numbers spoken in the video must be read from the block at the bottom of this file, which is
generated from `eval/results`.

## Shot list

| Time | Shot | Voice-over | On-screen text |
| --- | --- | --- | --- |
| 0:00–0:15 | Full-screen: original desk photo, cut to the Nano Banana edit. Then toggle **Difference** on the AI edit: the whole frame lights up. | "I asked Nano Banana for one thing: add an apple. I even said keep everything else unchanged. Here is what actually changed." | `One request.` then `Everything changed.` |
| 0:15–0:30 | Quick cuts: Banana100 paper title; the Android Authority and Adobe Community quotes. | "This is documented. Editors regenerate the whole frame on every edit. Across chained edits it compounds, and the standard quality metrics do not see it." | `Banana100, CVPR 2026 Workshops` · `"trimmed my beard and gave me a sunburn"` |
| 0:30–1:05 | Workbench. Click **Use your own images**, drop the two files, type the instruction, **Peel it**. The registration marks slide together; the change map wipes in. Then read the ledger: headline, drift score, alignment line, two region rows. Hover a row: its outline lights on the stage. | "Peel takes the original, the edit and the instruction. It aligns the two, measures every change, and works out which one you asked for. Cyan is kept. Magenta is peeled. Every change gets a sentence: the frame was shifted eight pixels, the plant was re-rendered, the lamp moved." | `Keep the edit.` `Peel the drift.` |
| 1:05–1:20 | Overlay **Off**. Move the loupe over the wall, the lamp, then the apple. Readout: "Peeled is identical to the original here". | "And this is not close enough. Outside the edit, the result is your original, byte for byte. The loupe counts the pixels." | `Byte-identical. Check it yourself.` |
| 1:20–1:55 | Second example, **A re-rendered face**: show the face region row and its identity distance, flip **Keep/Peel** once. Then the **Chains** page: filmstrip, raw vs peeled at the last step, the drift chart. | "Faces get an identity check: if the editor redrew a face you did not mention, it comes back. And in a chain, each step is anchored to the last peeled frame, so edit twenty still has the background of edit zero." | `Identity lock` · `Edit 20, background of edit 0` |
| 1:55–2:20 | **Numbers** page: the first table, then the ablation table, then the Banana100 chart. | Read two or three of the generated statements below. End with: "Existing diff-merge tools keep everything that changed. Peel knows which change was yours." | The table rows being read are highlighted with the cursor. |
| 2:20–2:35 | n8n: the workflow canvas, a form submission, the IF node taking the fail branch, the chat message with the ledger summary. | "For automated pipelines there is an n8n gate. An edit that touched the product does not ship; the repaired image and the reason do." | `n8n: drift gate` |
| 2:35–2:40 | Logo and URL on the gray stage, registration marks in register. | "Peel. Keep the edit. Peel the drift." | `Peel` · site URL · GitHub URL |

## Recording notes

- Use the bundled first example for 0:00–0:15 so nothing waits on the network.
- For 0:30–1:05 use the same pair through the upload panel (files in
  `web/public/examples/desk-apple/`: `original.webp` and `edited.webp`; instruction: "Add an apple on
  the table"). The run takes a few seconds; do not cut the progress stages, they are real.
- Set the browser zoom so the stage and the top of the ledger fit without scrolling.
- The API host stops idle instances after five minutes. Before recording, open https://peel-api-rahulgunwanistudy-5466s-projects.vercel.app/healthz and
  wait for `"status": "ok"`.
- On the public site uploads over about 4 MB in total are resized in the browser; for the video use
  the bundled 1024-px pair, or record against `make dev` for full resolution.
- Do not say any number that is not in the block below.

## Numbers that may be quoted

<!-- BEGIN NUMBERS -->
- On 216 synthetic cases with known ground truth, Peel restores 98.7% of unrequested pixel changes byte for byte; the diff-merge baseline restores 38.1%.
- Peel's kept region matches the requested edit with IoU 0.76 (diff-merge: 0.08), and unedited-region LPIPS drops from 0.164 in the raw edit to 0.028.
- Ablations: without instruction grounding the kept-region IoU falls from 0.76 to 0.49; without cast separation kept-edit fidelity falls from 28.3 to 27.7 dB (while IoU moves from 0.76 to 0.78); removing the background-locked re-registration changes IoU from 0.762 to 0.762 on this suite, where shifts are global and the first-pass fit already solves them.
- On a real Nano Banana chain from Banana100, after 20 chained edits the untouched areas of the raw chain are at LPIPS 0.388 and 17.5 dB PSNR against step 0; the peeled chain is at LPIPS 0.087 and 56.7 dB.
- When a re-rendered face touches the edit (24 cases), the face lock restores 90% of face pixels to the original (without it: 43%).
- End-to-end latency is 3.2 s at the median and 3.8 s at p95 on a 1024 × 558 px original with a 1024 × 547 px edit (peel-api-rahulgunwanistudy-5466s-projects.vercel.app (1 vCPU, Linux x86_64)), CPU only.
<!-- END NUMBERS -->
