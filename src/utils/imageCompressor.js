/**
 * Compresión y redimensionamiento de imágenes directamente en código.
 * Complejidad espacial: O(1) — reutiliza un único <canvas> offscreen pre-alocado
 * para evitar múltiples buffers en RAM al cargar fotos pesadas (4K / cámaras móviles).
 */

export const MAX_IMAGE_DIMENSION = 960
export const JPEG_QUALITY = 0.8

let sharedCanvas = null
let sharedCtx = null

function getSharedCanvas(width, height) {
  if (!sharedCanvas) {
    sharedCanvas = document.createElement('canvas')
    sharedCtx = sharedCanvas.getContext('2d', { alpha: false, willReadFrequently: false })
  }
  if (sharedCanvas.width !== width) sharedCanvas.width = width
  if (sharedCanvas.height !== height) sharedCanvas.height = height
  return { canvas: sharedCanvas, ctx: sharedCtx }
}

/**
 * Decodifica un File/Blob de imagen, lo escala proporcionalmente hasta `maxDim`
 * y lo comprime a JPEG en memoria liberando el bitmap original inmediatamente.
 *
 * @param {File|Blob} file Archivo de imagen original
 * @param {object} [opts]
 * @param {number} [opts.maxDim=960] Dimensión máxima en píxeles
 * @param {number} [opts.quality=0.8] Calidad JPEG (0..1)
 * @returns {Promise<{ url: string, width: number, height: number, originalBytes: number, compressedBytes: number }>}
 */
export async function compressImageFile(file, { maxDim = MAX_IMAGE_DIMENSION, quality = JPEG_QUALITY } = {}) {
  const originalBytes = file.size || 0

  let source = null
  let srcW = 0
  let srcH = 0
  let closeSource = () => {}

  if (typeof createImageBitmap === 'function') {
    const bitmap = await createImageBitmap(file)
    source = bitmap
    srcW = bitmap.width
    srcH = bitmap.height
    closeSource = () => bitmap.close()
  } else {
    const tempUrl = URL.createObjectURL(file)
    try {
      const img = await new Promise((resolve, reject) => {
        const el = new Image()
        el.onload = () => resolve(el)
        el.onerror = () => reject(new Error('No se pudo decodificar la imagen'))
        el.src = tempUrl
      })
      source = img
      srcW = img.naturalWidth || img.width
      srcH = img.naturalHeight || img.height
    } finally {
      URL.revokeObjectURL(tempUrl)
    }
  }

  try {
    const scale = Math.min(1, maxDim / Math.max(srcW, srcH))
    const targetW = Math.max(1, Math.round(srcW * scale))
    const targetH = Math.max(1, Math.round(srcH * scale))

    const { canvas, ctx } = getSharedCanvas(targetW, targetH)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'medium'
    ctx.drawImage(source, 0, 0, srcW, srcH, 0, 0, targetW, targetH)

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality))
    const outBlob = blob || file
    const url = URL.createObjectURL(outBlob)

    return {
      url,
      width: targetW,
      height: targetH,
      originalBytes,
      compressedBytes: outBlob.size || originalBytes,
    }
  } finally {
    closeSource()
  }
}

