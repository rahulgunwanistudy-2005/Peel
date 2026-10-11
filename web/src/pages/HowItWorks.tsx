import { useState } from 'react'

interface Step {
  title: string
  body: string
  image: string
  caption: string
}

const STEPS: Step[] = [
  {
    title: 'Register',
    body: 'Editors hand back a frame that is resized, shifted a pixel or two, or subtly warped. Peel matches SIFT keypoints, fits a homography with MAGSAC, then refits using only the background so the edit itself cannot drag the alignment. A dense optical-flow pass removes what is left.',
    image: './how/1-register.webp',
    caption: 'Difference from the original before alignment (left) and after (right).',
  },
  {
    title: 'Separate the colour cast',
    body: 'A slight tint over the whole frame would make every pixel look changed. Peel fits one colour transform to the stable pixels, reports it once in plain words, and then looks for local change on the corrected image.',
    image: './how/2-cast.webp',
    caption: 'Colour difference with the whole-image cast (left) and with it removed (right).',
  },
  {
    title: 'Measure change',
    body: 'Three views of difference are combined: CIEDE2000 for colour, gradient similarity for structure, and LPIPS for what a person would notice. The threshold adapts to the noise level of this particular pair, so a frame that was re-rendered everywhere does not drown the real changes.',
    image: './how/3-change.webp',
    caption: 'Change score. Brighter means more changed.',
  },
  {
    title: 'Locate what you asked for',
    body: 'The instruction is parsed into target phrases with plain rules, and each phrase is located with CLIPSeg in both images. Something you asked to add must light up in the edit but not in the original, which stops “red ball” from also selecting a red jacket.',
    image: './how/4-intent.webp',
    caption: 'Where the instruction points.',
  },
  {
    title: 'Check faces',
    body: 'Every face in the original is compared with the same face in the edit using a face-recognition embedding. If the identity moved and you did not ask for it, that face is restored, whatever else is going on around it.',
    image: './how/5-faces.webp',
    caption: 'Faces found in the original, with the identity distance to the edit.',
  },
  {
    title: 'Sort intended from collateral',
    body: 'Each changed region is compared with the instruction map. Mostly inside: intended, kept. Mostly outside: collateral, peeled. In between: flagged for you to review. A region that straddles the boundary is split so a stray change touching your edit can be peeled on its own.',
    image: './how/6-classify.webp',
    caption: 'Cyan is kept, magenta is peeled, yellow needs review.',
  },
  {
    title: 'Composite and write the ledger',
    body: 'Kept regions are feathered along real edges, tone-matched at the seam and blended with a Laplacian pyramid. Every other pixel is copied from your original, byte for byte. The ledger lists each change with its location, type and size, in a sentence.',
    image: './how/7-composite.webp',
    caption: 'Difference from the original: the raw AI edit (left) and the peeled result (right). Black is identical.',
  },
]

function Figure({ step }: { step: Step }) {
  const [missing, setMissing] = useState(false)
  if (missing) return null
  return (
    <figure>
      <img src={step.image} alt={step.caption} loading="lazy" onError={() => setMissing(true)} />
      <figcaption>{step.caption}</figcaption>
    </figure>
  )
}

export default function HowItWorks() {
  return (
    <div className="page">
      <header>
        <h1>How it works</h1>
        <p className="lede">
          Seven steps, all on a CPU, no generative model involved. Peel never invents a pixel: every pixel of the
          result comes from your original or from the edit you already have. The images below are real intermediate
          outputs from the first example on the workbench.
        </p>
      </header>
      <ol className="steps">
        {STEPS.map((step) => (
          <li key={step.title} className="step">
            <div>
              <h2>{step.title}</h2>
              <p>{step.body}</p>
            </div>
            <Figure step={step} />
          </li>
        ))}
      </ol>
    </div>
  )
}
