import { imageUrl, type Example } from '../api/client'

interface ExampleStripProps {
  examples: Example[]
  selected: string | null
  disabled: boolean
  onSelect: (example: Example) => void
  onOwn: () => void
}

export function ExampleStrip({ examples, selected, disabled, onSelect, onOwn }: ExampleStripProps) {
  return (
    <section className="examples" aria-label="Examples">
      <div className="examples-head">
        <h2>Try an example</h2>
        <button type="button" className="btn btn-solid" onClick={onOwn} disabled={disabled}>
          Use your own images
        </button>
      </div>
      <ul>
        {examples.map((example) => (
          <li key={example.id}>
            <button
              type="button"
              className="example"
              aria-pressed={selected === example.id}
              disabled={disabled}
              onClick={() => onSelect(example)}
            >
              <img
                src={imageUrl(example.result.images.edited, 320)}
                alt=""
                width={example.result.width}
                height={example.result.height}
                loading="lazy"
                decoding="async"
              />
              <span className="example-title">{example.title}</span>
              <span className="example-blurb">
                <q>{example.instruction}</q>
                {example.synthetic ? ' · synthetic drift' : ''}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
