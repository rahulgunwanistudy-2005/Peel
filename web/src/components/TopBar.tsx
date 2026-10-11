import { href, type Route } from '../lib/router'

const LINKS: { route: Route; label: string }[] = [
  { route: 'chain', label: 'Chains' },
  { route: 'how', label: 'How it works' },
  { route: 'numbers', label: 'Numbers' },
  { route: 'api', label: 'API' },
]

export const GITHUB_URL = 'https://github.com/rahulgunwanistudy-2005/Peel'

export function TopBar({ route }: { route: Route }) {
  return (
    <header className="bar">
      <a className="bar-name" href={href('workbench')} aria-current={route === 'workbench' ? 'page' : undefined}>
        <svg viewBox="0 0 28 28" width="22" height="22" aria-hidden="true">
          <circle cx="12" cy="14" r="7" fill="none" stroke="var(--cyan)" strokeWidth="2.2" />
          <circle cx="16" cy="14" r="7" fill="none" stroke="var(--magenta)" strokeWidth="2.2" />
          <path d="M14 2v24M2 14h24" stroke="var(--ink)" strokeWidth="1.5" />
        </svg>
        Peel
      </a>
      <p className="bar-line">Keep the edit you asked for. Undo everything else the AI changed.</p>
      <nav aria-label="Sections">
        <ul>
          {LINKS.map((link) => (
            <li key={link.route}>
              <a href={href(link.route)} aria-current={route === link.route ? 'page' : undefined}>
                {link.label}
              </a>
            </li>
          ))}
          <li>
            <a href={GITHUB_URL} rel="noreferrer">
              GitHub
            </a>
          </li>
        </ul>
      </nav>
    </header>
  )
}
