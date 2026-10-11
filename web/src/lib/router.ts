import { useEffect, useState } from 'react'

export type Route = 'workbench' | 'chain' | 'how' | 'numbers' | 'api'

const ROUTES: Route[] = ['workbench', 'chain', 'how', 'numbers', 'api']

function parse(hash: string): Route {
  const name = hash.replace(/^#\/?/, '').split(/[/?]/)[0]
  return (ROUTES as string[]).includes(name) ? (name as Route) : 'workbench'
}

/** Hash routing: the static build needs no server rewrites on Vercel or on the Space. */
export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash))
  useEffect(() => {
    const onChange = () => {
      setRoute(parse(window.location.hash))
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export function href(route: Route): string {
  return route === 'workbench' ? '#/' : `#/${route}`
}
