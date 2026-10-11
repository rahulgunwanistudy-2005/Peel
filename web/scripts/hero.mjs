// Capture frames for the README hero: the change map, then a compare sweep (AI edit vs peeled,
// difference view). Frames are assembled into a GIF by `python -m eval.hero`.
//   node scripts/hero.mjs [baseUrl] [framesDir]
import { chromium } from '@playwright/test'
import { mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'

const base = process.argv[2] ?? 'http://127.0.0.1:5173'
const out = resolve(process.argv[3] ?? '../eval/out/hero-frames')
rmSync(out, { recursive: true, force: true })
mkdirSync(out, { recursive: true })

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 })
await page.goto(`${base}/`)
await page.waitForLoadState('networkidle')
await page.waitForTimeout(2200)

let n = 0
const frame = async (repeat = 1) => {
  for (let i = 0; i < repeat; i += 1) {
    await page.screenshot({ path: `${out}/frame-${String(n).padStart(3, '0')}.png` })
    n += 1
  }
}

await frame(10) // change map and ledger
await page.getByRole('button', { name: 'Compare', exact: true }).click()
await page.getByRole('button', { name: 'Difference', exact: true }).click()
const slider = page.getByRole('slider')
for (const value of [50, 62, 74, 86, 96, 86, 74, 62, 50, 38, 26, 14, 4, 14, 26, 38, 50]) {
  await slider.fill(String(value))
  await page.waitForTimeout(60)
  await frame(value === 96 || value === 4 ? 5 : 1)
}
await page.getByRole('button', { name: 'Off', exact: true }).click()
await page.getByRole('button', { name: 'Peeled', exact: true }).click()
await page.waitForTimeout(200)
await frame(10) // the clean peeled result
await browser.close()
console.log(`wrote ${n} frames to ${out}`)
