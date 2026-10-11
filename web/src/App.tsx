import { lazy, Suspense } from 'react'

import { TopBar } from './components/TopBar'
import { useRoute } from './lib/router'
import { Workbench } from './pages/Workbench'

const Chain = lazy(() => import('./pages/Chain'))
const HowItWorks = lazy(() => import('./pages/HowItWorks'))
const Numbers = lazy(() => import('./pages/Numbers'))
const ApiPage = lazy(() => import('./pages/ApiPage'))

export default function App() {
  const route = useRoute()
  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <TopBar route={route} />
      <main id="main">
        {route === 'workbench' ? (
          <Workbench />
        ) : (
          <Suspense fallback={<p className="page-loading">Loading</p>}>
            {route === 'chain' && <Chain />}
            {route === 'how' && <HowItWorks />}
            {route === 'numbers' && <Numbers />}
            {route === 'api' && <ApiPage />}
          </Suspense>
        )}
      </main>
    </>
  )
}
