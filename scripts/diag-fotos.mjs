// Diagnóstico de fotos con el modelo real (sin navegador).
// 1) powershell -File scripts/letterbox.ps1 -Src <carpeta fotos> -Dst D:\tmp\diag
// 2) node scripts/diag-fotos.mjs D:\tmp\diag
// Muestra, por foto, la decisión de la app (umbrales de classes.json) y todas las cabezas
// que el modelo ve con confianza >= 0.15, para distinguir "confianza baja" de "no ve nada".
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'
import { postprocess, evaluateAccess, setInputSize, setScoreThresholds } from '../src/utils/inferenceEngine.js'

const dir = process.argv[2] || 'D:/tmp/diag'
const iou = Number(process.argv[3]) || undefined // opcional: umbral de solapamiento del NMS
const modelDir = 'public/model_heads'

try {
  await tf.setBackend('wasm')
} catch {
  await tf.setBackend('cpu')
}
await tf.ready()

const meta = JSON.parse(readFileSync(join(modelDir, 'classes.json'), 'utf8'))
const classes = Object.keys(meta.classes).sort((a, b) => a - b).map((k) => meta.classes[k])
const size = meta.imgsz
setInputSize(size)
if (meta.thresholds) setScoreThresholds(meta.thresholds)
const boxScale = meta.box_scale?.startsWith('normalized') ? 'normalized_0_1' : 'pixels'

const mj = JSON.parse(readFileSync(join(modelDir, 'model.json'), 'utf8'))
const weightSpecs = mj.weightsManifest.flatMap((g) => g.weights)
const parts = mj.weightsManifest.flatMap((g) => g.paths).map((p) => readFileSync(join(modelDir, p)))
const all = Buffer.concat(parts)
const weightData = all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength)
const model = await tf.loadGraphModel(tf.io.fromMemory({ modelTopology: mj.modelTopology, weightSpecs, weightData }))

const SHORT = { con_casco: 'casco', sin_casco: 'SIN casco' }
let photos = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8').replace(/^\uFEFF/, ''))
if (!Array.isArray(photos)) photos = [photos]

console.log(`backend=${tf.getBackend()} imgsz=${size} iou=${iou ?? 'por defecto'}\n`)
for (const p of photos) {
  const bytes = readFileSync(join(dir, p.file))
  const input = tf.tidy(() => tf.tensor3d(new Uint8Array(bytes), [size, size, 3]).toFloat().div(255).expandDims(0))
  const t0 = performance.now()
  const out = model.execute(input)
  const o = Array.isArray(out) ? out[0] : out
  const params = { scale: p.scale, padX: p.padX, padY: p.padY, srcW: p.srcW, srcH: p.srcH }
  const app = await postprocess(o, params, classes, { boxScale, iou })
  const low = await postprocess(o, params, classes, { boxScale, iou, thresholds: { helmet: 0.15, noHelmet: 0.15 } })
  const ms = Math.round(performance.now() - t0)
  tf.dispose([input, ...(Array.isArray(out) ? out : [out])])

  const r = evaluateAccess(app)
  const verdict = { ALLOWED: 'PERMITE', DENIED: 'DENIEGA', IDLE: 'NADA' }[r.status]
  console.log(`${p.name}  ->  ${verdict}  (con casco ${r.withHelmetCount}, sin casco ${r.withoutHelmetCount})  ${ms} ms`)
  for (const d of low) {
    const h = Math.round(((d.y2 - d.y1) * p.scale))
    const used = app.some((a) => Math.abs(a.x1 - d.x1) < 2 && Math.abs(a.y1 - d.y1) < 2)
    console.log(`   ${used ? '*' : ' '} ${(SHORT[d.className] ?? d.className).padEnd(9)} ${d.score.toFixed(2)}  x=${Math.round(d.x1)} y=${Math.round(d.y1)}  alto en entrada=${h}px`)
  }
  console.log('')
}
console.log('* = cuenta para la decisión de la app')
