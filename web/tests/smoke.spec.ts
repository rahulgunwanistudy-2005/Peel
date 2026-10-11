import { expect, test } from '@playwright/test'

test('load an example, toggle views, flip a region', async ({ page }) => {
  await page.goto('/')

  // The first view already shows a peeled example and its ledger.
  const ledger = page.getByRole('complementary', { name: 'Drift ledger' })
  await expect(ledger.getByRole('heading', { name: 'Drift ledger' })).toBeVisible()
  await expect(ledger.getByRole('meter', { name: 'Drift score' })).toBeVisible()
  const stageImage = page.locator('.stage-frame img.layer').first()
  await expect(stageImage).toBeVisible()
  await expect(stageImage).toHaveAttribute('src', /peeled/)

  // Toggle views.
  await page.getByRole('button', { name: 'Original', exact: true }).click()
  await expect(stageImage).toHaveAttribute('src', /original/)
  await page.getByRole('button', { name: 'AI edit', exact: true }).click()
  await expect(stageImage).toHaveAttribute('src', /edited/)
  await page.getByRole('button', { name: 'Compare', exact: true }).click()
  await expect(page.getByRole('slider')).toBeVisible()
  await page.getByRole('button', { name: 'Peeled', exact: true }).click()
  await page.getByRole('button', { name: 'Off', exact: true }).click()
  await expect(page.locator('.stage-frame img.overlay')).toHaveCount(0)

  // The loupe reports pixel identity where nothing was kept (top-left corner of the frame).
  const frame = page.locator('.stage-frame')
  const box = await frame.boundingBox()
  if (!box) throw new Error('stage frame has no box')
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.08)
  await expect(page.locator('.loupe-readout')).toContainText('Peeled is identical to the original here')
  await page.mouse.move(0, 0)

  // Switch example, then flip the kept region: the server recomposites and the ledger follows.
  await page.getByRole('button', { name: /re-rendered face/i }).click()
  await expect(ledger).toContainText('Add a yellow balloon')
  const kept = ledger.locator('.region', { hasText: 'The change you asked for' })
  await expect(kept.locator('.region-state')).toContainText('Kept')
  await kept.getByRole('button', { name: 'Peel', exact: true }).click()
  await expect(kept.locator('.region-state')).toContainText('Peeled by you')
  await expect(ledger).toContainText('100.0% of the result is byte-for-byte your original')
  await expect(stageImage).toHaveAttribute('src', /\/v1\/jobs\/ex-astronaut-face\/images\/peeled\.png\?v=1/)

  // Flip it back.
  await kept.getByRole('button', { name: 'Keep', exact: true }).click()
  await expect(kept.locator('.region-state')).toContainText('Kept')
  await expect(kept.locator('.region-state')).not.toContainText('by you')
})

test('other pages render', async ({ page }) => {
  for (const [path, heading] of [
    ['#/how', 'How it works'],
    ['#/numbers', 'Numbers'],
    ['#/api', 'API'],
    ['#/chain', 'Chains'],
  ]) {
    await page.goto(`/${path}`)
    await expect(page.getByRole('heading', { level: 1, name: heading })).toBeVisible()
  }
})

test('unrelated images get a specific, human error', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Use your own images' }).click()
  const inputs = page.locator('input[type=file]')
  await inputs.nth(0).setInputFiles('public/examples/astronaut-face/original.webp')
  await inputs.nth(1).setInputFiles('public/examples/coffee-recolor/original.webp')
  await page.getByLabel('What you asked the editor to do').fill('add a hat')
  await page.getByRole('button', { name: 'Peel it' }).click()
  await expect(page.getByRole('alert')).toContainText('These look like two different photos')
})

test('upload a pair, follow the stages, get a ledger', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Use your own images' }).click()
  const inputs = page.locator('input[type=file]')
  await inputs.nth(0).setInputFiles('public/examples/desk-apple/original.webp')
  await inputs.nth(1).setInputFiles('public/examples/desk-apple/edited.webp')
  await page.getByLabel('What you asked the editor to do').fill('Add an apple on the table')
  await page.getByRole('button', { name: 'Peel it' }).click()

  // Real stage names arrive over Server-Sent Events while the job runs.
  await expect(page.locator('.progress')).toBeVisible()
  const ledger = page.getByRole('complementary', { name: 'Drift ledger' })
  await expect(ledger).toContainText('Add an apple on the table', { timeout: 90_000 })
  await expect(ledger).toContainText('Kept 1 requested change')
  await expect(ledger).toContainText('Analysed in')
  await expect(page.locator('.stage-frame img.layer').first()).toHaveAttribute('src', /\/v1\/jobs\/[0-9a-f]+\/images\/peeled\.png/)
  await expect(page.locator('.marks')).toHaveAttribute('data-phase', 'registered')
  await expect(page.locator('.stage-frame img.overlay')).toHaveAttribute('src', /change_map/)
})
