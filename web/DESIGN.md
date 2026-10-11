# Peel — design plan

Two passes, as the frontend-design skill asks: a first plan, a review against the brief and the
skill's list of generic defaults, then the revised plan that was built.

## Pass 1 — first plan

- **Concept.** The print-proofing desk. Peel does what a prepress operator does: put two sheets in
  register, separate what belongs from what does not, sign off the proof. The vocabulary of that
  desk (registration marks, a colour bar, a loupe, a job ticket) maps one-to-one onto the product.
- **Colour.** Mid-gray stage, light chrome, CMY inks for the three decisions.
- **Type.** One grotesk, tabular figures for every number.
- **Layout.** Stage on the left (two thirds), ledger on the right (one third).
- **Motion.** Registration marks slide into register, then the change map wipes in.

## Review of pass 1

Checked against the brief and against the things that make an interface look generated:

| Risk in pass 1 | Decision |
| --- | --- |
| "Light chrome" drifts to off-white/cream, which tints colour judgment and is a stock look. | Chrome is strictly neutral: R = G = B on every surface. No warm or cool grays anywhere. |
| A ledger of rounded cards with soft shadows. | No cards, no shadows, no radius above 2 px. Rows are separated by tone and space, like lines on a job ticket. |
| Hairline rules everywhere (the "newspaper" tell). | Rules only where a real ticket has them: under the ticket head, and between regions. Everything else is tonal blocks. |
| Ink colours leaking into buttons, links and hover states as decoration. | Inks appear in exactly four places: region overlays, ledger swatches, the drift meter, the registration marks. Interactive chrome is black on gray. |
| A hero number with a gradient. | The drift score is plain black tabular figures on the ticket, next to a densitometer-style scale. |
| Uppercase eyebrow labels and monospace micro-labels for "technical" flavour. | Sentence case throughout, one family, no monospace except real code on the API page. |
| Motion sprinkled on hover. | One orchestrated moment (register, then wipe). Everything else is still. Focus and hover change tone, not position. |
| White text on magenta fails AA (4.3:1). | Ink is never a text background. A swatch carries the colour; the label next to it is black on gray. |
| Loupe as a generic circular zoom lens. | A linen tester: square, three panes side by side (original, AI edit, peeled) locked to the same spot, with a pixel-identity readout. It is the proof of the bit-identical claim, so it gets real estate. |

## Pass 2 — the plan that was built

### Principles
1. **The image is the subject.** Chrome never competes with it: neutral, flat, quiet.
2. **Colour means a decision.** Cyan = kept, magenta = peeled, yellow = review. Nothing else is coloured.
3. **Every claim is checkable on screen.** The loupe and the difference view let a judge verify "byte-identical outside the edit" themselves.
4. **One motion moment.** Register, then reveal.
5. **Plain words.** Each region reads as a sentence a non-expert understands; numbers follow, never lead.

### Colour tokens
| Token | Value | Use |
| --- | --- | --- |
| `--stage` | `#777777` | Image stage. L* = 50: the standard neutral surround for judging colour. A lighter, darker or tinted surround shifts perceived brightness and hue of the image (simultaneous contrast), so this is functional, not taste. |
| `--desk` | `#D9D9D9` | Page background around the stage and ticket. |
| `--ticket` | `#EDEDED` | Ledger and panels. |
| `--ticket-2` | `#E3E3E3` | Hovered/selected row, input wells. |
| `--rule` | `#BDBDBD` | The few rules. |
| `--ink` | `#141414` | Text, icons, buttons (the K of CMYK). 14.9:1 on `--ticket`. |
| `--ink-2` | `#4A4A4A` | Secondary text. 7.6:1 on `--ticket`. |
| `--cyan` | `#00AEEF` | Kept. |
| `--magenta` | `#EC008C` | Peeled. |
| `--yellow` | `#FFDD00` | Review. |
| `--on-stage` | `#FFFFFF` | Marks and labels drawn on the stage (4.5:1 on `#777` needs white at bold/large; small labels sit on an `--ink` chip). |

Inks are process cyan, magenta and yellow; the fourth ink, black, is the type.

### Type
- **Schibsted Grotesk** (variable, self-hosted, 45 kB). A sturdy news grotesk with open counters and
  true tabular figures behind `tnum` (verified in the font's GSUB table; default figures are
  proportional). Hanken Grotesk was rejected because it has no `tnum` feature.
- Scale (px): 12 · 13 · 15 · 18 · 24 · 40. Body 15/22. Weights 400, 500, 700.
- `font-variant-numeric: tabular-nums` on every element that shows a measurement.
- Sentence case everywhere. No letter-spaced capitals.

### Layout
- Slim top bar: name, one sentence, four links.
- Workbench: stage (fluid, about two thirds) and ledger ticket (fixed 400 px column). Below 900 px
  they stack, stage first. Works to 375 px.
- Stage has a slug line underneath (view, overlay, zoom readout), like the slug of a proof.
- Image boxes reserve their aspect ratio before load: no layout shift.
- 4 px spacing grid; 2 px corner radius at most.

### Signature motion
On every run (and once on first load), each corner of the stage shows two crosshairs, cyan and
magenta, out of register. During "Registering" they slide together into one black mark; then the
change map wipes across. About 1.4 s in total. With `prefers-reduced-motion` the marks are simply
drawn in register and the map appears without the wipe.

### Components
Top bar · stage (view switch, compare slider, overlays, region outlines) · linen-tester loupe ·
ledger ticket (drift score and scale, cast, registration, byte-identical share, region rows with
before/after crops and a keep/peel switch, warnings, timings, downloads) · upload panel · example
strip · chain filmstrip and drift chart · tables for the numbers page.

### Accessibility
Keyboard reachable everything, visible 2 px black focus ring with a white halo so it shows on the
stage too, real `button`/`label`/`input` elements, alt text on every image, the compare slider is
an `input type="range"`, state is never colour-only (swatches always have a word next to them).

## Screenshot critique (after building)

Screenshots were taken with Playwright (`scripts/screenshots.mjs`, 1440 × 900 and 375 × 812) and
reviewed against the plan. What was wrong, and what was changed:

| Seen in the screenshot | Change |
| --- | --- |
| "8 . 4" and "93 . 5%": the font's `tnum` feature also widens the full stop to a digit. | A `Num` component sets tabular figures on digits and normal spacing on the punctuation between them. |
| Compare + Difference showed the plain AI edit on the left and the difference on the right: apples and oranges. | In that mode the left half shows the raw edit's difference and the right half the peeled result's, so the drift visibly disappears as the divider moves. |
| Five rows in a row read "A patch … changed colour" for thin slivers along the lamp. | The classifier now recognises slivers as moved edges; the sentence says "redrawn or moved". |
| Timings on a bundled example reported the laptop that precomputed it. | Bundled examples say they were analysed ahead of time; only live runs show timings. |
| Scrolling the ledger moved the image out of view, so hovering a region far down the list highlighted nothing visible. | On wide screens the stage is sticky while the ledger scrolls. |
| The step numbers on "How it works" inherited tabular figures and looked spaced out. | Proportional figures in headings. |

Checked and left as they are: the inks appear only on overlays, swatches, the drift scale and the
registration marks; every surface is a neutral gray; nothing has a shadow or a radius above 2 px;
the mobile layout stacks stage, examples, ledger in that order with the loupe docked under the
stage.

Known compromises: the registration marks are low-contrast as cyan and magenta on the gray stage
(they are decorative until they register into a black mark, which is 4.2:1), and the loupe's
floating panel covers part of the image next to the cursor on narrow desktop windows.
