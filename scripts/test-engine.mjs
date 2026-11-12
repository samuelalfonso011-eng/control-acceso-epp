// Prueba del post-procesado con un tensor sintético (sin navegador): node scripts/test-engine.mjs
// Modelo por cabezas: salida [1, 6, 3549] (4 caja + 2 clases) a 416 px.
import * as tf from '@tensorflow/tfjs'
import assert from 'node:assert/strict'
import { isSoftwareRenderer } from '../src/utils/backend.js'
import { postprocess, evaluateAccess, setScoreThresholds, getScoreThresholds } from '../src/utils/inferenceEngine.js'

await tf.setBackend('cpu')
await tf.ready()

const classes = ['con_casco', 'sin_casco']
const OK = 0 // con_casco
const BAD = 1 // sin_casco
const CH = 4 + classes.length // 6 canales
const N = 3549 // 52² + 26² + 13² anclas a 416
const params = { scale: 1, padX: 0, padY: 0, srcW: 416, srcH: 416 }

// Construye [1, 6, n] con las anclas dadas: {i, cx, cy, w, h, cls, score}
function makeOutput(anchors, n = N) {
  const data = new Float32Array(CH * n)
  for (const a of anchors) {
    data[0 * n + a.i] = a.cx
    data[1 * n + a.i] = a.cy
    data[2 * n + a.i] = a.w
    data[3 * n + a.i] = a.h
    data[(4 + a.cls) * n + a.i] = a.score
  }
  return tf.tensor(data, [1, CH, n])
}

const before = tf.memory().numTensors

// Caso 1: conductor con casco + copiloto sin casco (+ duplicado del primero que el NMS debe eliminar)
let out = makeOutput([
  { i: 10, cx: 150, cy: 100, w: 40, h: 40, cls: OK, score: 0.94 },
  { i: 11, cx: 152, cy: 101, w: 40, h: 40, cls: OK, score: 0.9 }, // duplicado
  { i: 20, cx: 210, cy: 110, w: 40, h: 40, cls: BAD, score: 0.91 },
])
let det = await postprocess(out, params, classes)
out.dispose()
assert.equal(det.length, 2, 'NMS debe dejar 2 cabezas')
let r = evaluateAccess(det)
assert.equal(r.status, 'DENIED')
assert.equal(r.withHelmetCount, 1)
assert.equal(r.withoutHelmetCount, 1)
const helmet = det.find((d) => d.hasHelmet)
assert.deepEqual([helmet.x1, helmet.y1, helmet.x2, helmet.y2].map(Math.round), [130, 80, 170, 120])

// Caso 2: dos con casco -> ALLOWED
out = makeOutput([
  { i: 1, cx: 150, cy: 150, w: 40, h: 40, cls: OK, score: 0.8 },
  { i: 2, cx: 250, cy: 160, w: 40, h: 40, cls: OK, score: 0.7 },
])
det = await postprocess(out, params, classes)
out.dispose()
r = evaluateAccess(det)
assert.equal(r.status, 'ALLOWED')
assert.equal(r.withHelmetCount, 2)

// Caso 3: cabezas pegadas (conductor y copiloto) que se solapan poco -> se conservan las dos
out = makeOutput([
  { i: 1, cx: 200, cy: 150, w: 40, h: 40, cls: OK, score: 0.8 },
  { i: 2, cx: 230, cy: 150, w: 40, h: 40, cls: BAD, score: 0.7 }, // IoU = 10/70 ≈ 0.14 < 0.45
])
det = await postprocess(out, params, classes)
out.dispose()
assert.equal(det.length, 2)

// Caso 4: moto llena (9 cabezas sin casco) -> se cuentan todas
out = makeOutput(Array.from({ length: 9 }, (_, k) => ({ i: k, cx: 30 + k * 42, cy: 200, w: 36, h: 36, cls: BAD, score: 0.6 })))
det = await postprocess(out, params, classes)
out.dispose()
assert.equal(evaluateAccess(det).withoutHelmetCount, 9)

// Bajo umbral -> IDLE
out = makeOutput([{ i: 5, cx: 200, cy: 200, w: 40, h: 40, cls: OK, score: 0.3 }])
det = await postprocess(out, params, classes)
out.dispose()
assert.equal(evaluateAccess(det).status, 'IDLE')

// Una clase desconocida en classes.json se ignora (el mapeo es por nombre)
out = makeOutput([{ i: 5, cx: 200, cy: 200, w: 40, h: 40, cls: OK, score: 0.9 }])
det = await postprocess(out, params, ['otra_cosa', 'sin_casco'])
out.dispose()
assert.equal(det.length, 0)

// Des-letterbox: fuente 1280x720 a 416 -> scale .325, padY 91
out = makeOutput([{ i: 7, cx: 208, cy: 208, w: 65, h: 65, cls: OK, score: 0.9 }])
det = await postprocess(out, { scale: 0.325, padX: 0, padY: 91, srcW: 1280, srcH: 720 }, classes)
out.dispose()
assert.deepEqual([det[0].x1, det[0].y1, det[0].x2, det[0].y2].map(Math.round), [540, 260, 740, 460])

// El número de anclas se lee de la salida (p. ej. un export a 320: [1, 6, 2100])
out = makeOutput([{ i: 42, cx: 160, cy: 160, w: 50, h: 50, cls: BAD, score: 0.88 }], 2100)
det = await postprocess(out, { scale: 0.25, padX: 0, padY: 70, srcW: 1280, srcH: 720 }, classes)
out.dispose()
assert.equal(det.length, 1)
assert.equal(det[0].className, 'sin_casco')
assert.deepEqual([det[0].x1, det[0].y1, det[0].x2, det[0].y2].map(Math.round), [540, 260, 740, 460])

// --- Umbral asimétrico (por defecto helmet 0.50 / noHelmet 0.30) ---
const run = async (anchors, options) => {
  const o = makeOutput(anchors)
  const d = await postprocess(o, params, classes, options)
  o.dispose()
  return d
}
const box = { i: 5, cx: 200, cy: 200, w: 40, h: 40 }
assert.deepEqual(getScoreThresholds(), { helmet: 0.5, noHelmet: 0.3 })

// sin casco a 0.35 -> cuenta -> DENIED
det = await run([{ ...box, cls: BAD, score: 0.35 }])
assert.equal(evaluateAccess(det).status, 'DENIED')
// sin casco a 0.29 -> bajo el umbral -> IDLE
det = await run([{ ...box, cls: BAD, score: 0.29 }])
assert.equal(evaluateAccess(det).status, 'IDLE')
// casco a 0.45 no cuenta (umbral de casco intacto) -> IDLE
det = await run([{ ...box, cls: OK, score: 0.45 }])
assert.equal(evaluateAccess(det).status, 'IDLE')
// misma ancla: casco 0.45 (no llega) + sin casco 0.35 (sí llega) -> sin casco
{
  const o = makeOutput([{ ...box, cls: OK, score: 0.45 }])
  o.bufferSync().set(0.35, 0, 4 + BAD, box.i)
  det = await postprocess(o, params, classes)
  o.dispose()
  assert.equal(det.length, 1)
  assert.equal(det[0].className, 'sin_casco')
  assert.ok(Math.abs(det[0].score - 0.35) < 1e-6)
}
// misma ancla: casco 0.90 + sin casco 0.35 -> gana el casco (no se niega a quien claramente lo lleva)
{
  const o = makeOutput([{ ...box, cls: OK, score: 0.9 }])
  o.bufferSync().set(0.35, 0, 4 + BAD, box.i)
  det = await postprocess(o, params, classes)
  o.dispose()
  assert.equal(evaluateAccess(det).status, 'ALLOWED')
}
// copiloto débil (0.35) junto a conductor con casco -> DENIED
det = await run([
  { i: 1, cx: 150, cy: 150, w: 40, h: 40, cls: OK, score: 0.83 },
  { i: 2, cx: 250, cy: 150, w: 40, h: 40, cls: BAD, score: 0.35 },
])
assert.equal(evaluateAccess(det).status, 'DENIED')

// Configurable: por llamada...
det = await run([{ ...box, cls: BAD, score: 0.35 }], { thresholds: { noHelmet: 0.4 } })
assert.equal(det.length, 0)
det = await run([{ ...box, cls: OK, score: 0.45 }], { thresholds: { helmet: 0.4 } })
assert.equal(evaluateAccess(det).status, 'ALLOWED')
// ...y global (lo que hace useHelmetModel con classes.json)
setScoreThresholds({ noHelmet: 0.4 })
det = await run([{ ...box, cls: BAD, score: 0.35 }])
assert.equal(det.length, 0)
assert.deepEqual(setScoreThresholds(), { helmet: 0.5, noHelmet: 0.3 }) // sin argumento: restablece
assert.throws(() => setScoreThresholds({ noHelmet: 1.5 }))
assert.throws(() => setScoreThresholds({ helmet: '0.5' }))
assert.deepEqual(getScoreThresholds(), { helmet: 0.5, noHelmet: 0.3 }) // un valor inválido no cambia nada

// Detección de WebGL por software (sin GPU real -> se usa WASM directo)
for (const r of ['ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)), SwiftShader driver)', 'Microsoft Basic Render Driver', 'llvmpipe (LLVM 15.0.7, 256 bits)', 'Software Rasterizer']) {
  assert.equal(isSoftwareRenderer(r), true, r)
}
for (const r of ['ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0, D3D11)', 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1650 Direct3D11)', 'Apple M2', '', null]) {
  assert.equal(isSoftwareRenderer(r), false, String(r))
}

// Sin fugas de tensores
assert.equal(tf.memory().numTensors, before, 'fuga de tensores')

console.log('OK: postprocess (2 clases, 6 canales) + evaluateAccess + des-letterbox + sin fugas')
