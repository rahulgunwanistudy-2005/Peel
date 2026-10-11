// Capture the screenshots used for design review and for the Devpost page.
//   node scripts/screenshots.mjs [baseUrl] [outDir]
import { chromium } from '@playwright/test'
import { mkdirSync } from 'node:fs'
import { resolve } from 'node:path'

const base = process.argv[2] ?? 'http://127.0.0.1:5173'
const out = resolve(process.argv[3] ?? '../docs/screenshots')
mkdirSync(out, { recursive: true })

const browser = await chromium.launch()

async function shot(name, { width, height, path = '#/', fullPage = false, prepare } = {}) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 2 })
  const page = await context.newPage()
  await page.goto(`${base}/${path}`)
  await page.waitForLoadState('networkidle')
  await page.waitForTimeout(1800)
  if (prepare) await prepare(page)
  await page.screenshot({ path: `${out}/${name}.png`, fullPage })
  await context.close()
  console.log('wrote', name)
}

await shot('01-workbench', { width: 1440, height: 900 })
await shot('02-workbench-full', { width: 1440, height: 900, fullPage: true })
await shot('03-loupe', {
  width: 1440,
  height: 900,
  prepare: async (page) => {
    await page.getByRole('button', { name: 'Off', exact: true }).click()
    const frame = page.locator('.stage-frame')
    const box = await frame.boundingBox()
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.45)
    await page.waitForTimeout(900)
  },
})
await shot('04-compare', {
  width: 1440,
  height: 900,
  prepare: async (page) => {
    await page.getByRole('button', { name: 'Compare', exact: true }).click()
    await page.getByRole('button', { name: 'Difference', exact: true }).click()
    await page.waitForTimeout(500)
  },
})
await shot('05-face-example', {
  width: 1440,
  height: 900,
  prepare: async (page) => {
    await page.getByRole('button', { name: /re-rendered face/i }).click()
    await page.waitForTimeout(2000)
  },
})
await shot('06-upload', {
  width: 1440,
  height: 900,
  fullPage: true,
  prepare: async (page) => {
    await page.getByRole('button', { name: 'Use your own images' }).click()
    await page.waitForTimeout(300)
  },
})
await shot('07-chain', { width: 1440, height: 900, path: '#/chain', fullPage: true })
await shot('08-numbers', { width: 1440, height: 900, path: '#/numbers', fullPage: true })
await shot('09-how', { width: 1440, height: 900, path: '#/how', fullPage: true })
await shot('10-api', { width: 1440, height: 900, path: '#/api', fullPage: true })
await shot('11-mobile', { width: 375, height: 812, fullPage: true })
await browser.close()
