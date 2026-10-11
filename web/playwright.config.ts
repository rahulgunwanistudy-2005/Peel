import { defineConfig } from '@playwright/test'

const API_PORT = 7860
const WEB_PORT = 5173

/**
 * The smoke test runs against the real API and the real web app. Both servers are started here
 * unless they are already running.
 */
const REMOTE = process.env.PEEL_E2E_BASE // run the same tests against a deployed site

export default defineConfig({
  testDir: './tests',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  reporter: [['list']],
  use: { baseURL: REMOTE ?? `http://127.0.0.1:${WEB_PORT}`, viewport: { width: 1440, height: 900 } },
  webServer: REMOTE ? undefined : [
    {
      command: `../.venv/bin/python -m uvicorn app.main:app --app-dir ../api --host 127.0.0.1 --port ${API_PORT}`,
      url: `http://127.0.0.1:${API_PORT}/healthz`,
      reuseExistingServer: true,
      timeout: 180_000,
    },
    {
      command: `npm run dev -- --host 127.0.0.1 --port ${WEB_PORT} --strictPort`,
      url: `http://127.0.0.1:${WEB_PORT}`,
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
})
