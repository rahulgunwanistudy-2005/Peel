/**
 * Some hosts cap the size of a whole request. When the chosen images do not fit, they are scaled
 * down in the browser (all to the same long side, losslessly re-encoded as PNG) until they do.
 */
export interface Fitted {
  files: Blob[]
  /** Long side in pixels the images were reduced to, or null when they were sent untouched. */
  resizedTo: number | null
}

async function scaled(file: Blob, longSide: number): Promise<Blob> {
  const bitmap = await createImageBitmap(file)
  const scale = Math.min(1, longSide / Math.max(bitmap.width, bitmap.height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(bitmap.width * scale))
  canvas.height = Math.max(1, Math.round(bitmap.height * scale))
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('no 2d context')
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  bitmap.close()
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('could not encode'))), 'image/png')
  })
}

export async function fitImages(files: Blob[], maxTotalBytes: number, startSide: number): Promise<Fitted> {
  const total = files.reduce((sum, f) => sum + f.size, 0)
  if (maxTotalBytes <= 0 || total <= maxTotalBytes) return { files, resizedTo: null }
  let side = startSide
  for (;;) {
    const resized = await Promise.all(files.map((f) => scaled(f, side)))
    if (resized.reduce((sum, f) => sum + f.size, 0) <= maxTotalBytes || side <= 384) {
      return { files: resized, resizedTo: side }
    }
    side = Math.round(side * 0.82)
  }
}
