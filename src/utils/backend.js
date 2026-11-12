import * as tf from '@tensorflow/tfjs'

// Los .wasm y el script UMD del backend se copian a public/tfjs-wasm/ desde node_modules
// (plugin en vite.config.js).
const WASM_PATH = '/tfjs-wasm/'
const WASM_SCRIPT = 'tf-backend-wasm.es2017.min.js' // ES2017: extiende clases nativas de tf (el ES5 no puede)
const WASM_INIT_TIMEOUT_MS = 20000
const SOFTWARE_GL = /basic render|swiftshader|llvmpipe|software/i
const BENCH_RUNS = 3

/** true si el renderer de WebGL es un rasterizador por software (sin GPU real). */
export function isSoftwareRenderer(renderer) {
  return SOFTWARE_GL.test(String(renderer || ''))
}

/** Lee WEBGL_debug_renderer_info sin inicializar TensorFlow.js. */
export function probeGpu() {
  const none = { available: false, renderer: null, software: false }
  if (typeof document === 'undefined') return none
  try {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl')
    if (!gl) return none
    const ext = gl.getExtension('WEBGL_debug_renderer_info')
    const renderer = String((ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)) || '')
    gl.getExtension('WEBGL_lose_context')?.loseContext()
    return { available: true, renderer, software: isSoftwareRenderer(renderer) }
  } catch {
    return none
  }
}

/**
 * Permite forzar un backend desde la URL para comparar tiempos: ?backend=webgl | wasm | cpu
 * (sin parámetro, se elige midiendo).
 */
export function forcedBackend() {
  if (typeof location === 'undefined') return null
  const value = new URLSearchParams(location.search).get('backend')
  return ['webgl', 'wasm', 'cpu'].includes(value) ? value : null
}

let wasmApi = null
let wasmLoading = null

/**
 * Carga el backend WASM desde su script UMD servido tal cual (NO empaquetado).
 * Motivo: con hilos, el backend crea sus workers serializando una función con toString();
 * si esa función pasa por el minificador del bundler, el worker falla
 * ("ReferenceError: e is not defined") y la inicialización se queda colgada.
 * El UMD registra el backend 'wasm' en el mismo motor de TensorFlow.js de la app (window.tf).
 */
function loadWasmBackend() {
  if (wasmApi) return Promise.resolve(wasmApi)
  if (wasmLoading) return wasmLoading
  wasmLoading = new Promise((resolve, reject) => {
    // Copia plana del namespace: el UMD necesita poder escribir tf.wasm
    if (!window.tf) window.tf = { ...tf }
    const script = document.createElement('script')
    script.src = WASM_PATH + WASM_SCRIPT
    script.onload = () => {
      wasmApi = window.tf.wasm
      if (!wasmApi?.setWasmPaths) return reject(new Error('El script del backend WASM no se registró'))
      wasmApi.setWasmPaths(WASM_PATH)
      resolve(wasmApi)
    }
    script.onerror = () => reject(new Error(`No se pudo cargar ${WASM_PATH}${WASM_SCRIPT}`))
    document.head.appendChild(script)
  }).catch((err) => {
    wasmLoading = null
    throw err
  })
  return wasmLoading
}

const withTimeout = (promise, ms, what) =>
  Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${what}: sin respuesta tras ${ms / 1000} s`)), ms)),
  ])

/** Activa un backend ('webgl' | 'wasm' | 'cpu'). Lanza si no se puede inicializar. */
export async function activateBackend(name) {
  if (name === 'wasm') {
    await loadWasmBackend()
    // Si los workers fallan, la inicialización nunca termina: el límite evita una carga eterna.
    const ok = await withTimeout(tf.setBackend('wasm'), WASM_INIT_TIMEOUT_MS, 'Backend WASM')
    if (!ok) throw new Error('No se pudo inicializar el backend wasm')
  } else {
    const ok = await tf.setBackend(name)
    if (!ok) throw new Error(`No se pudo inicializar el backend ${name}`)
  }
  await tf.ready()
  return name
}

/**
 * Backend inicial para poder cargar el modelo:
 * con GPU real se intenta webgl primero; con WebGL por software (o sin WebGL) se va directo a wasm.
 * No se fijan flags de WebGL: FORCE_F16_TEXTURES y DELETE_TEXTURE_THRESHOLD=0 no aportaban
 * y el segundo obligaba a recrear texturas en cada frame.
 */
export async function initBackend() {
  const gpu = probeGpu()
  const forced = forcedBackend()
  if (forced) {
    // Sin respaldo a propósito: si el backend pedido no arranca, el error debe verse.
    await activateBackend(forced)
    return { backend: forced, gpu, realGpu: false, forced: true }
  }
  const realGpu = gpu.available && !gpu.software
  const order = realGpu ? ['webgl', 'wasm', 'cpu'] : ['wasm', 'cpu']
  let lastErr = null
  for (const name of order) {
    try {
      await activateBackend(name)
      return { backend: name, gpu, realGpu }
    } catch (err) {
      lastErr = err
      console.warn(`[backend] ${name} no disponible:`, err?.message || err)
    }
  }
  throw lastErr || new Error('Ningún backend de TensorFlow.js disponible')
}

async function runOnce(model, imgsz) {
  const x = tf.zeros([1, imgsz, imgsz, 3])
  const t0 = performance.now()
  const res = typeof model.execute === 'function' ? model.execute(x) : await model.executeAsync(x)
  const outs = Array.isArray(res) ? res : [res]
  await outs[0].data()
  const ms = performance.now() - t0
  tf.dispose([x, ...outs])
  return ms
}

/** Warm-up + mediana de BENCH_RUNS inferencias en el backend indicado (ms). */
export async function benchmarkBackend(name, model, imgsz, runs = BENCH_RUNS) {
  await activateBackend(name)
  await runOnce(model, imgsz) // warm-up: compila shaders (webgl) / reserva memoria (wasm)
  const times = []
  for (let i = 0; i < runs; i++) times.push(await runOnce(model, imgsz))
  times.sort((a, b) => a - b)
  return times[Math.floor(times.length / 2)]
}

/**
 * Elige el backend definitivo MIDIENDO:
 * - con GPU real: mide webgl y wasm y se queda con el más rápido;
 * - sin GPU real: mide solo el backend activo (wasm, o cpu si wasm falló).
 * El modelo ya debe estar cargado; sus pesos se mueven solos entre backends.
 */
export async function selectFastestBackend(model, imgsz, init) {
  const candidates = init.realGpu ? ['webgl', 'wasm'] : [init.backend]
  const benchmarks = {}
  for (const name of candidates) {
    try {
      benchmarks[name] = await benchmarkBackend(name, model, imgsz)
    } catch (err) {
      console.warn(`[backend] no se pudo medir ${name}:`, err?.message || err)
    }
  }

  let backend = Object.keys(benchmarks).sort((a, b) => benchmarks[a] - benchmarks[b])[0]
  if (!backend) {
    // Nada se pudo medir: último recurso, cpu
    backend = 'cpu'
    benchmarks.cpu = await benchmarkBackend('cpu', model, imgsz, 1)
  }
  if (tf.getBackend() !== backend) await activateBackend(backend)

  const info = {
    backend,
    forced: init.forced === true,
    renderer: init.gpu.renderer,
    softwareGL: init.gpu.software,
    benchmarks,
    isolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true,
    simd: null,
    threads: null,
  }
  if (backend === 'wasm') {
    info.simd = await tf.env().getAsync('WASM_HAS_SIMD_SUPPORT')
    const multi = await tf.env().getAsync('WASM_HAS_MULTITHREAD_SUPPORT')
    info.threads = multi && wasmApi?.getThreadsCount ? wasmApi.getThreadsCount() : 1
  }
  console.info('[backend] elegido:', info)
  return info
}

/** Texto corto para la interfaz, p. ej. "SIMD · 4 hilos". */
export function describeBackend(info) {
  if (!info) return ''
  if (info.backend === 'wasm') {
    const parts = [info.simd ? 'SIMD' : 'sin SIMD', info.threads > 1 ? `${info.threads} hilos` : '1 hilo']
    if (!info.isolated) parts.push('sin aislamiento COOP/COEP')
    return parts.join(' · ')
  }
  if (info.backend === 'webgl') return info.renderer ? info.renderer.replace(/^ANGLE \(|\)$/g, '').slice(0, 60) : 'GPU'
  return 'JavaScript puro (lento)'
}
