import { Fragment } from 'react'

/**
 * Tabular figures for the digits, normal spacing for the punctuation between them. The font's
 * `tnum` feature also widens the full stop to a digit's width, which reads as "8 . 4".
 */
export function Num({ children }: { children: string }) {
  const parts = children.split(/([.,:])/)
  return (
    <span className="num">
      {parts.map((part, index) =>
        part === '.' || part === ',' || part === ':' ? (
          <span key={index} className="num-point">
            {part}
          </span>
        ) : (
          <Fragment key={index}>{part}</Fragment>
        ),
      )}
    </span>
  )
}
