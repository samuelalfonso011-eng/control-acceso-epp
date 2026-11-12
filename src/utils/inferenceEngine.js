import * as tf from '@tensorflow/tfjs'
import { HELMET_OK, HELMET_BAD } from './classMap.js'

// Tamaño de entrada del modelo: lo fija useHelmetModel con `imgsz` de classes.json (320 o 640).
export let INPUT_SIZE = 640
export const NUM_ANCHORS = 8400 // solo respaldo: el real se lee de output.shape[2] (2100 a 320, 8400 a 640)

export function setInputSize(size) {
  if (!Number.isInteger(size) || size <= 0 || size % 32 !== 0) {
    throw new Error(`imgsz inválido en classes.json: ${size} (debe ser múltiplo de 32)`)
  }
  INPUT_SIZE = size
  if (letterboxCanvas) {
    letterboxCanvas.width = size
    letterboxCanvas.height = size
  }
}
export const IOU_THRESHOLD = 0.45

/**
 * Umbral ASIMÉTRICO de confianza (ante la duda, se niega el acceso):
 *  - helmet:   mínimo para contar a alguien como CON casco.
 *  - noHelmet: mínimo para contar a alguien como SIN casco (sin_casco).
 * Tres formas de configurarlo, de menor a mayor prioridad:
 *  1) estos valores por defecto;
 *  2) el campo opcional "thresholds": { "helmet": 0.5, "noHelmet": 0.3 } en public/model_heads/classes.json
 *     (viaja con cada modelo; lo aplica useHelmetModel con setScoreThresholds);
 *  3) por llamada: detect(model, classes, source, { thresholds: { noHelmet: 0.25 } }).
 */
export const DEFAULT_SCORE_THRESHOLDS = Object.freeze({ helmet: 0.5, noHelmet: 0.3 })
let scoreThresholds = { ...DEFAULT_SCORE_THRESHOLDS }

function mergeThresholds(base, patch) {
  const out = { ...base }
  for (const key of ['helmet', 'noHelmet']) {
    const v = patch?.[key]
    if (v === undefined || v === null) continue
    if (typeof v !== 'number' || !(v > 0 && v < 1)) {
      throw new Error(`Umbral "${key}" inválido: ${v} (debe ser un número entre 0 y 1)`)
    }
    out[key] = v
  }
  return out
}

/** Fija los umbrales globales partiendo de los valores por defecto. Sin argumento, los restablece. */
export function setScoreThresholds(patch) {
  scoreThresholds = mergeThresholds(DEFAULT_SCORE_THRESHOLDS, patch)
  return { ...scoreThresholds }
}

export function getScoreThresholds() {
  return { ...scoreThresholds }
}
export const MAX_DETECTIONS = 20 // una caja por cabeza: margen para motos llenas y gente alrededor

const INV_255 = 1 / 255

export const ACCESS = { IDLE: 'IDLE', ALLOWED: 'ALLOWED', DENIED: 'DENIED' }

export const ACCESS_MESSAGES = {
  IDLE: 'Esperando motociclista en zona de control...',
  ALLOWED: 'Acceso permitido, bienvenido',
  DENIED: 'Acceso denegado, por favor ponerse el casco',
}

const COLOR_OK = '#10B981'
const COLOR_BAD = '#EF4444'
const LABELS = {
  con_casco: 'Casco',
  sin_casco: 'Sin Casco',
}

// Buffer de letterbox pre-alocado una sola vez -> O(1) en memoria de pre-procesamiento
let letterboxCanvas = null
let letterboxCtx = null

function getLetterboxBuffer() {
  if (!letterboxCanvas && typeof document !== 'undefined') {
    letterboxCanvas = document.createElement('canvas')
    letterboxCanvas.width = INPUT_SIZE
    letterboxCanvas.height = INPUT_SIZE
    letterboxCtx = letterboxCanvas.getContext('2d', { alpha: false, willReadFrequently: false })
  }
  return letterboxCtx
}

/** Índices de canal (dentro de las clases) que cuentan como ocupante; cualquier otra clase se descarta. */
function relevantClassIds(classes) {
  const ids = []
  classes.forEach((name, i) => {
    if (HELMET_OK.includes(name) || HELMET_BAD.includes(name)) ids.push(i)
  })
  return ids
}

/**
 * Pre-procesamiento con compresión espacial en canvas pre-alocado (O(1) memoria auxiliar):
 * En navegador usa el motor 2D nativo para escalar y aplicar letterbox a INPUT_SIZE² antes de
 * subir la textura a WebGL, evitando alocar tensores intermedios de alta resolución (resizeBilinear + pad).
 */
export function preprocess(source) {
  const w = source.videoWidth || source.naturalWidth || source.width
  const h = source.videoHeight || source.naturalHeight || source.height
  if (!w || !h) throw new Error('La fuente visual aún no tiene dimensiones')

  const scale = Math.min(INPUT_SIZE / w, INPUT_SIZE / h)
  const newW = Math.round(w * scale)
  const newH = Math.round(h * scale)
  const padX = Math.floor((INPUT_SIZE - newW) / 2)
  const padY = Math.floor((INPUT_SIZE - newH) / 2)

  const ctx = getLetterboxBuffer()

  const input = tf.tidy(() => {
    if (ctx) {
      ctx.fillStyle = 'rgb(114, 114, 114)'
      ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE)
      ctx.drawImage(source, 0, 0, w, h, padX, padY, newW, newH)
      return tf.browser.fromPixels(letterboxCanvas).toFloat().mul(INV_255).expandDims(0)
    }

    // Fallback para entornos sin DOM (ej. pruebas unitarias en Node)
    const img = tf.browser.fromPixels(source)
    const resized = tf.image.resizeBilinear(img, [newH, newW])
    const padded = tf.pad(
      resized,
      [
        [padY, INPUT_SIZE - newH - padY],
        [padX, INPUT_SIZE - newW - padX],
        [0, 0],
      ],
      114,
    )
    return padded.mul(INV_255).expandDims(0)
  })

  return { input, params: { scale, padX, padY, srcW: w, srcH: h } }
}

/** IoU en O(1) entre dos cajas { x1, y1, x2, y2 }. */
function computeIoU(a, b) {
  const xx1 = Math.max(a.x1, b.x1)
  const yy1 = Math.max(a.y1, b.y1)
  const xx2 = Math.min(a.x2, b.x2)
  const yy2 = Math.min(a.y2, b.y2)
  const w = Math.max(0, xx2 - xx1)
  const h = Math.max(0, yy2 - yy1)
  if (w === 0 || h === 0) return 0
  const inter = w * h
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
  const union = areaA + areaB - inter
  return union > 0 ? inter / union : 0
}

/**
 * Post-procesado YOLOv8 en tiempo O(N) y memoria auxiliar O(1) (cero tensores intermedios en GPU):
 * - Lee el buffer plano [1, 4+nc, 8400] en una sola transferencia GPU->CPU.
 * - Descarta en O(1) por ancla todas las que no superen su umbral (>99.5% del total)
 *   sin transponer la matriz ni crear 18 tensores en WebGL.
 * - Umbral asimétrico por ancla:
 *     a) si la mejor clase supera SU umbral (helmet o noHelmet), se acepta con esa clase;
 *     b) si no, pero alguna clase SIN casco supera `noHelmet`, se acepta como sin casco
 *        (ej.: casco 0.45 / sin casco 0.35 -> cuenta como sin casco: ante la duda, se niega).
 * - Ejecuta NMS agnóstico a la clase únicamente sobre las K candidatas sobrevivientes (K << N).
 */
export async function postprocess(output, params, classes, { boxScale = 'pixels', thresholds, iou = IOU_THRESHOLD } = {}) {
  const ids = relevantClassIds(classes)
  if (!ids.length) return []
  const { helmet: thrHelmet, noHelmet: thrNoHelmet } = mergeThresholds(scoreThresholds, thresholds)
  const isBad = ids.map((id) => HELMET_BAD.includes(classes[id]))

  const raw = await output.data()
  const numAnchors = output.shape[2] || NUM_ANCHORS
  const kScale = boxScale === 'normalized_0_1' ? INPUT_SIZE : 1
  const numRelevant = ids.length

  const candidates = []

  // Recorrido lineal O(N) con acceso directo por stride (c * numAnchors + i)
  for (let i = 0; i < numAnchors; i++) {
    let bestScore = -1
    let bestClassId = -1
    let bestIsBad = false
    let badScore = -1 // mejor puntaje entre las clases SIN casco
    let badClassId = -1

    for (let c = 0; c < numRelevant; c++) {
      const classId = ids[c]
      const score = raw[(4 + classId) * numAnchors + i]
      if (score > bestScore) {
        bestScore = score
        bestClassId = classId
        bestIsBad = isBad[c]
      }
      if (isBad[c] && score > badScore) {
        badScore = score
        badClassId = classId
      }
    }

    if (bestScore < (bestIsBad ? thrNoHelmet : thrHelmet)) {
      // La mejor clase (casco) no alcanza su umbral: ¿hay evidencia suficiente de "sin casco"?
      if (bestIsBad || badScore < thrNoHelmet) continue
      bestScore = badScore
      bestClassId = badClassId
    }

    const xc = raw[i] * kScale
    const yc = raw[numAnchors + i] * kScale
    const bw = raw[2 * numAnchors + i] * kScale
    const bh = raw[3 * numAnchors + i] * kScale
    const halfW = bw * 0.5
    const halfH = bh * 0.5

    candidates.push({
      x1: xc - halfW,
      y1: yc - halfH,
      x2: xc + halfW,
      y2: yc + halfH,
      score: bestScore,
      classId: bestClassId,
    })
  }

  if (!candidates.length) return []

  // Ordenar candidatas por confianza descendente (K suele ser < 15)
  candidates.sort((a, b) => b.score - a.score)

  // NMS voraz sobre las K candidatas sobrevivientes
  const selected = []
  for (let i = 0; i < candidates.length && selected.length < MAX_DETECTIONS; i++) {
    const cand = candidates[i]
    let keep = true
    for (let j = 0; j < selected.length; j++) {
      if (computeIoU(cand, selected[j]) > iou) {
        keep = false
        break
      }
    }
    if (keep) selected.push(cand)
  }

  const { scale, padX, padY, srcW, srcH } = params
  const invScale = 1 / scale
  const clamp = (v, max) => Math.min(Math.max(v, 0), max)

  return selected.map((d) => {
    const className = classes[d.classId]
    return {
      x1: clamp((d.x1 - padX) * invScale, srcW),
      y1: clamp((d.y1 - padY) * invScale, srcH),
      x2: clamp((d.x2 - padX) * invScale, srcW),
      y2: clamp((d.y2 - padY) * invScale, srcH),
      score: d.score,
      className,
      hasHelmet: HELMET_OK.includes(className),
    }
  })
}

/** Regla de negocio: todos los ocupantes deben llevar casco. */
export function evaluateAccess(detections) {
  let withHelmetCount = 0
  for (let i = 0; i < detections.length; i++) {
    if (detections[i].hasHelmet) withHelmetCount++
  }
  const withoutHelmetCount = detections.length - withHelmetCount
  let status = ACCESS.IDLE
  if (withoutHelmetCount > 0) status = ACCESS.DENIED
  else if (withHelmetCount > 0) status = ACCESS.ALLOWED
  return {
    status,
    message: ACCESS_MESSAGES[status],
    withHelmetCount,
    withoutHelmetCount,
    total: detections.length,
  }
}

/**
 * Pipeline completo: fuente visual -> { detections, result, latencyMs }.
 * Libera el tensor de entrada y el de salida en cada llamada (O(1) tensores en VRAM).
 */
export async function detect(model, classes, source, options) {
  const t0 = performance.now()
  const { input, params } = preprocess(source)
  let out
  try {
    const res = typeof model.execute === 'function' ? model.execute(input) : await model.executeAsync(input)
    out = Array.isArray(res) ? res : [res]
    const detections = await postprocess(out[0], params, classes, options)
    return {
      detections,
      result: evaluateAccess(detections),
      latencyMs: performance.now() - t0,
    }
  } finally {
    tf.dispose(input)
    if (out) tf.dispose(out)
  }
}

/** Dibuja cajas y etiquetas. El canvas toma el tamaño de la fuente; el CSS lo escala. */
export function drawDetections(canvas, detections, width, height) {
  if (!canvas) return
  if (canvas.width !== width) canvas.width = width
  if (canvas.height !== height) canvas.height = height
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, width, height)
  if (!detections.length) return

  // Cajas de cabeza (pequeñas): trazo y etiqueta más finos para no tapar la cara ni la caja vecina
  const lw = Math.max(2, Math.round(width / 280))
  const fontPx = Math.max(12, Math.round(width / 55))
  ctx.lineWidth = lw
  ctx.font = `bold ${fontPx}px system-ui, sans-serif`
  ctx.textBaseline = 'top'

  for (let i = 0; i < detections.length; i++) {
    const d = detections[i]
    const color = d.hasHelmet ? COLOR_OK : COLOR_BAD
    const label = `${LABELS[d.className] ?? d.className}: ${Math.round(d.score * 100)}%`
    const bw = d.x2 - d.x1
    const bh = d.y2 - d.y1

    ctx.strokeStyle = color
    ctx.strokeRect(d.x1, d.y1, bw, bh)

    const pad = Math.round(fontPx * 0.3)
    const tw = ctx.measureText(label).width + pad * 2
    const th = fontPx + pad * 2
    const ty = d.y1 - th >= 0 ? d.y1 - th : d.y1
    ctx.fillStyle = color
    ctx.fillRect(d.x1, ty, tw, th)
    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, d.x1 + pad, ty + pad)
  }
}
