import { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { setInputSize, setScoreThresholds } from '../utils/inferenceEngine.js'
import { initBackend, selectFastestBackend, activateBackend } from '../utils/backend.js'

export { HELMET_OK, HELMET_BAD } from '../utils/classMap.js'

const MODEL_URL = '/model_heads/model.json'
const CLASSES_URL = '/model_heads/classes.json'
// Sube MODEL_VERSION cada vez que reemplaces el modelo en public/model_heads con el MISMO imgsz
// (p. ej. tras re-entrenar); si no, el navegador seguiría usando la copia vieja de IndexedDB.
// La clave incluye arquitectura (`model` en classes.json) e imgsz para no mezclar exports,
// y el prefijo "heads" la separa del modelo anterior (rider_*), que era también yolov8n-416.
const MODEL_VERSION = 1
const idbModelUrl = (imgsz, arch) => `indexeddb://helmet-heads-${arch}-${imgsz}-v${MODEL_VERSION}`

// Singleton en memoria (O(1)): evita doble descarga y doble warm-up en React StrictMode
let cachedResult = null
let bootPromise = null
const progressListeners = new Set()

function notifyProgress(p) {
  progressListeners.forEach((fn) => fn(p))
}

async function loadModelWithCache(imgsz, arch) {
  const IDB_MODEL_URL = idbModelUrl(imgsz, arch)
  // 1. Intenta cargar desde IndexedDB (evita releer ~12.5 MB desde la USB en recargas)
  try {
    const cachedModel = await tf.loadGraphModel(IDB_MODEL_URL)
    notifyProgress(0.9)
    return cachedModel
  } catch {
    // No está en caché aún; carga desde MODEL_URL y guarda en IndexedDB en segundo plano
  }

  const netModel = await tf.loadGraphModel(MODEL_URL, {
    onProgress: (p) => notifyProgress(0.05 + p * 0.85),
  })

  netModel.save(IDB_MODEL_URL).catch(() => {
    /* ignorar si el navegador restringe IndexedDB en modo incógnito */
  })

  return netModel
}

async function bootSingleton() {
  if (cachedResult) return cachedResult
  if (bootPromise) return bootPromise

  bootPromise = (async () => {
    const [init, meta] = await Promise.all([
      initBackend(),
      fetch(CLASSES_URL).then((res) => {
        if (!res.ok) throw new Error(`No se pudo cargar ${CLASSES_URL} (${res.status})`)
        return res.json()
      }),
    ])
    notifyProgress(0.05)

    const classes = Object.entries(meta.classes)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([, name]) => name)

    const imgsz = Number(meta.imgsz) || 640
    setInputSize(imgsz)
    // Umbrales por modelo: campo opcional "thresholds" en classes.json; si falta, valores por defecto.
    const thresholds = setScoreThresholds(meta.thresholds)
    console.info(`[modelo] ${meta.model || 'yolov8n'} @${imgsz} · umbrales`, thresholds)

    const model = await loadModelWithCache(imgsz, meta.model || 'yolov8n')

    // El modelo debe esperar el mismo tamaño que declara classes.json
    const inShape = model.inputs?.[0]?.shape
    if (inShape && inShape[1] > 0 && inShape[1] !== imgsz) {
      throw new Error(
        `classes.json dice imgsz=${imgsz} pero model.json espera ${inShape[1]}x${inShape[2]}. ` +
          'Copia ambos archivos del mismo export.',
      )
    }

    // Warm-up + medición: con GPU real compara webgl y wasm y se queda con el más rápido;
    // sin GPU real mide el backend activo. Deja los shaders/memoria listos antes del primer frame.
    notifyProgress(0.92)
    const backendInfo = await selectFastestBackend(model, imgsz, init)

    notifyProgress(1)
    cachedResult = { backend: backendInfo.backend, backendInfo, model, classes }
    // Solo en desarrollo: acceso desde la consola para diagnosticar puntajes crudos y medir backends
    if (import.meta.env.DEV) window.__helmet = { model, classes, tf, backendInfo, activateBackend }
    return cachedResult
  })()

  try {
    return await bootPromise
  } catch (err) {
    bootPromise = null
    throw err
  }
}

/**
 * Ciclo de vida del modelo: backend -> classes.json -> GraphModel (con caché IndexedDB) -> warm-up.
 * Devuelve { loading, progress, backend, backendInfo, modelReady, error, model, classes }.
 */
export function useHelmetModel() {
  const [state, setState] = useState(() =>
    cachedResult
      ? {
          loading: false,
          progress: 1,
          backend: cachedResult.backend,
          backendInfo: cachedResult.backendInfo,
          modelReady: true,
          error: null,
          model: cachedResult.model,
          classes: cachedResult.classes,
        }
      : {
          loading: true,
          progress: 0,
          backend: null,
          backendInfo: null,
          modelReady: false,
          error: null,
          model: null,
          classes: null,
        },
  )

  useEffect(() => {
    if (cachedResult) return
    let cancelled = false
    const onProgress = (progress) => {
      if (!cancelled) setState((s) => ({ ...s, progress }))
    }
    progressListeners.add(onProgress)

    bootSingleton()
      .then(({ backend, backendInfo, model, classes }) => {
        if (!cancelled) {
          setState({
            loading: false,
            progress: 1,
            backend,
            backendInfo,
            modelReady: true,
            error: null,
            model,
            classes,
          })
        }
      })
      .catch((e) => {
        console.error(e)
        if (!cancelled) {
          setState((s) => ({ ...s, error: e.message || String(e), loading: false }))
        }
      })

    return () => {
      cancelled = true
      progressListeners.delete(onProgress)
    }
  }, [])

  return state
}
