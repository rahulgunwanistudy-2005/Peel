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
- If the API is asleep, open `/healthz` first and wait for `"status": "ok"`.
- Do not say any number that is not in the block below.

## Numbers that may be quoted

<!-- BEGIN NUMBERS -->
- Not measured yet.
<!-- END NUMBERS -->
