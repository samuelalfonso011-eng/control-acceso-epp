import { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Camera, CameraOff, Cpu, Database, Gauge, RefreshCw, Timer, Wifi } from 'lucide-react'
import { detect, drawDetections } from '../utils/inferenceEngine.js'
import { createStabilizer } from '../utils/stabilizer.js'
import { describeBackend } from '../utils/backend.js'

function Stat({ icon: Icon, label, value, sub }) {
  return (
    <div className="min-w-0 rounded-xl bg-slate-800 px-4 py-3">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
        <Icon className="size-4 shrink-0" aria-hidden /> {label}
      </div>
      <div className="mt-1 text-2xl font-bold tabular-nums break-words text-white">{value}</div>
      {sub && <div className="mt-0.5 text-xs break-words text-slate-400">{sub}</div>}
    </div>
  )
}

/**
 * Abre el stream de cámara con cascada de restricciones:
 * 1) 640x480 (resolución nativa óptima para YOLOv8 640x640, ahorra ancho de banda y RAM)
 * 2) 1280x720
 * 3) { video: true } universal (compatible con cualquier webcam USB o integrada en Windows)
 */
async function openCameraStream(deviceId) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('El navegador no expone navigator.mediaDevices.getUserMedia (la página debe abrirse por HTTPS).')
  }

  const attempts = deviceId
    ? [
        { video: { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }, audio: false },
        { video: { deviceId: { exact: deviceId } }, audio: false },
      ]
    : [
        { video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } }, audio: false },
        { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
        { video: true, audio: false },
      ]

  let lastErr = null
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints)
    } catch (err) {
      lastErr = err
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        throw err
      }
    }
  }
  throw lastErr || new Error('No se pudo inicializar ninguna cámara de video.')
}

const SLOW_INFERENCE_MS = 1000 // por encima de esto se considera equipo lento
const SLOW_FRAME_STRIDE = 5 // en equipo lento: analizar 1 de cada 5 frames

export default function LiveCamera({ model, classes, backend, backendInfo, onAccess }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const rafRef = useRef(0)
  const sessionRef = useRef(0)
  const inferringRef = useRef(false)
  const startingRef = useRef(false)
  const stabilizerRef = useRef(createStabilizer(350))

  const modelRef = useRef(model)
  const classesRef = useRef(classes)
  const onAccessRef = useRef(onAccess)
  modelRef.current = model
  classesRef.current = classes
  onAccessRef.current = onAccess

  const [active, setActive] = useState(false)
  const [starting, setStarting] = useState(false)
  const [devices, setDevices] = useState([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [error, setError] = useState(null)
  const [telemetry, setTelemetry] = useState({ latency: 0, fps: 0, tensors: 0, vramMb: 0, resolution: '—', slow: false })

  const refreshDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return
      const all = await navigator.mediaDevices.enumerateDevices()
      const cams = all.filter((d) => d.kind === 'videoinput')
      setDevices(cams)
    } catch {
      /* ignorar si aún no hay permisos */
    }
  }, [])

  const stopCamera = useCallback(() => {
    sessionRef.current++ // invalida el bucle de inferencia en curso
    cancelAnimationFrame(rafRef.current)
    rafRef.current = 0
    inferringRef.current = false
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.srcObject = null
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    setActive(false)
    setStarting(false)
    startingRef.current = false
    onAccessRef.current(stabilizerRef.current.reset())
  }, [])

  const startCamera = useCallback(
    async (deviceIdOverride) => {
      if (startingRef.current) return
      startingRef.current = true
      setStarting(true)
      setError(null)

      // Cierra cualquier stream previo antes de abrir uno nuevo
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }

      try {
        const targetDevice = deviceIdOverride ?? selectedDeviceId
        const stream = await openCameraStream(targetDevice || undefined)
        streamRef.current = stream

        const video = videoRef.current
        if (!video) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }

        video.srcObject = stream

        // Espera a que el video tenga dimensiones válidas antes de reproducir
        if (video.readyState < 1) {
          await new Promise((resolve) => {
            const onLoaded = () => {
              video.removeEventListener('loadedmetadata', onLoaded)
              resolve()
            }
            video.addEventListener('loadedmetadata', onLoaded, { once: true })
            setTimeout(resolve, 1500)
          })
        }

        try {
          await video.play()
        } catch (playErr) {
          // Ignora AbortError benigno si el navegador ya está reproduciendo el stream
          if (playErr.name !== 'AbortError') throw playErr
        }

        await refreshDevices()
        const activeTrack = stream.getVideoTracks()[0]
        const settings = activeTrack?.getSettings?.() || {}
        if (settings.deviceId && !selectedDeviceId) {
          setSelectedDeviceId(settings.deviceId)
        }

        setActive(true)
        setStarting(false)
        startingRef.current = false

        // --- Bucle de inferencia SECUENCIAL ---
        // El <video> lo pinta el navegador a la tasa de la cámara por su cuenta; aquí solo se
        // actualiza el canvas superpuesto con la ÚLTIMA detección. La siguiente inferencia se
        // lanza únicamente cuando terminó la anterior (nunca una por frame ni con setInterval).
        const session = ++sessionRef.current
        const alive = () => sessionRef.current === session && streamRef.current === stream

        // Espera al siguiente frame NUEVO de la cámara (o al siguiente repintado si no hay rVFC)
        const nextFrame = () =>
          new Promise((resolve) => {
            const v = videoRef.current
            if (v && typeof v.requestVideoFrameCallback === 'function') v.requestVideoFrameCallback(() => resolve())
            else rafRef.current = requestAnimationFrame(() => resolve())
          })

        const loop = async () => {
          let lastResultTime = performance.now()
          let lastUiUpdate = 0
          let fpsEma = 0
          let latEma = 0
          let framesToSkip = 0

          while (alive()) {
            await nextFrame()
            if (!alive()) break
            if (framesToSkip > 0) {
              framesToSkip--
              continue
            }

            const v = videoRef.current
            const m = modelRef.current
            const cls = classesRef.current
            if (!(v && m && cls && v.readyState >= 2 && v.videoWidth > 0)) continue

            try {
              const { detections, result, latencyMs } = await detect(m, cls, v)
              if (!alive()) break

              drawDetections(canvasRef.current, detections, v.videoWidth, v.videoHeight)
              onAccessRef.current(stabilizerRef.current.push(result, performance.now()))

              const now = performance.now()
              const instFps = 1000 / Math.max(now - lastResultTime, 1)
              lastResultTime = now
              fpsEma = fpsEma ? fpsEma * 0.85 + instFps * 0.15 : instFps
              latEma = latEma ? latEma * 0.85 + latencyMs * 0.15 : latencyMs

              // Equipo lento (> 1 s por inferencia): se analiza un frame de cada SLOW_FRAME_STRIDE
              // para que la interfaz respire entre cálculos.
              const slow = latEma > SLOW_INFERENCE_MS
              framesToSkip = slow ? SLOW_FRAME_STRIDE - 1 : 0

              if (now - lastUiUpdate > 300) {
                lastUiUpdate = now
                const mem = tf.memory()
                setTelemetry({
                  latency: latEma,
                  fps: fpsEma,
                  tensors: mem.numTensors,
                  vramMb: mem.numBytes / (1024 * 1024),
                  resolution: `${v.videoWidth}×${v.videoHeight}`,
                  slow,
                })
              }
            } catch (err) {
              console.error('Error en inferencia de cámara:', err)
              framesToSkip = 30 // evita un bucle cerrado de errores
            }
          }
        }

        loop()
      } catch (e) {
        console.error('Error al abrir la cámara:', e)
        let msg = `No se pudo abrir la cámara: ${e.message || e.name}`
        if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
          msg = 'Permiso de cámara denegado. Habilita el acceso a la cámara en la barra de direcciones del navegador.'
        } else if (e.name === 'NotFoundError' || e.name === 'DevicesNotFoundError') {
          msg = 'No se detectó ninguna cámara conectada a este equipo.'
        } else if (e.name === 'NotReadableError' || e.name === 'TrackStartError') {
          msg = 'La cámara está siendo usada por otra aplicación (Zoom, Teams, otra pestaña). Ciérrala y vuelve a intentar.'
        }
        setError(msg)
        setActive(false)
        setStarting(false)
        startingRef.current = false
      }
    },
    [refreshDevices, selectedDeviceId],
  )

  useEffect(() => {
    refreshDevices()
    return () => {
      stopCamera()
    }
  }, [refreshDevices, stopCamera])

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_290px]">
      <div className="flex min-w-0 flex-col gap-3">
        <div className="relative aspect-video overflow-hidden rounded-2xl bg-black ring-1 ring-slate-700">
          <video ref={videoRef} playsInline muted autoPlay className="absolute inset-0 size-full object-contain" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full object-contain" />

          {!active && (
            <div className="absolute inset-0 grid place-items-center bg-slate-950/60 p-3 text-center sm:p-4">
              <div className="flex flex-col items-center gap-3">
                <button
                  onClick={() => startCamera()}
                  disabled={starting}
                  className="flex cursor-pointer items-center gap-2 rounded-2xl bg-sky-500 px-5 py-3 text-base font-bold text-slate-950 hover:bg-sky-400 disabled:opacity-60 sm:gap-3 sm:px-8 sm:py-4 sm:text-xl"
                >
                  <Camera className="size-6 shrink-0 sm:size-7" />
                  {starting ? 'Iniciando cámara…' : 'Activar cámara en vivo'}
                </button>
                <span className="hidden text-sm text-slate-300 sm:block">
                  Captura comprimida a 640×480 · Inferencia O(N) en un solo paso
                </span>
              </div>
            </div>
          )}

          {active && (
            <div className="absolute right-2 top-2 flex items-center gap-2 sm:right-3 sm:top-3">
              <button
                onClick={stopCamera}
                className="flex cursor-pointer items-center gap-2 rounded-xl bg-slate-900/85 px-3 py-2 text-sm font-semibold sm:px-4 sm:text-base text-white ring-1 ring-slate-700 hover:bg-red-600"
              >
                <CameraOff className="size-5" /> Detener
              </button>
            </div>
          )}
        </div>

        {devices.length > 1 && (
          <div className="flex flex-wrap items-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 ring-1 ring-slate-800">
            <label htmlFor="camera-select" className="text-sm font-semibold text-slate-300">
              Dispositivo de video:
            </label>
            <select
              id="camera-select"
              value={selectedDeviceId}
              onChange={(e) => {
                const id = e.target.value
                setSelectedDeviceId(id)
                if (active) startCamera(id)
              }}
              className="min-w-0 max-w-full flex-1 truncate rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-white ring-1 ring-slate-700 sm:flex-none"
            >
              {devices.map((d, idx) => (
                <option key={d.deviceId || idx} value={d.deviceId}>
                  {d.label || `Cámara ${idx + 1}`}
                </option>
              ))}
            </select>
            <button
              onClick={refreshDevices}
              title="Actualizar lista de cámaras"
              aria-label="Actualizar lista de cámaras"
              className="shrink-0 cursor-pointer rounded-lg bg-slate-800 p-2.5 text-slate-300 hover:bg-slate-700"
            >
              <RefreshCw className="size-4" />
            </button>
          </div>
        )}
      </div>

      <aside className="grid min-w-0 grid-cols-1 content-start gap-3 min-[480px]:grid-cols-2 lg:grid-cols-1" aria-label="Telemetría Edge AI">
        <h3 className="col-span-full text-sm font-bold uppercase tracking-wide text-slate-400">Telemetría Edge AI · O(1) VRAM</h3>
        <Stat
          icon={Timer}
          label="Latencia de inferencia"
          value={`${telemetry.latency.toFixed(0)} ms`}
          sub={`Stream: ${telemetry.resolution}`}
        />
        <Stat
          icon={Gauge}
          label="FPS de análisis"
          value={telemetry.fps.toFixed(1)}
          sub={telemetry.slow ? `Equipo lento: 1 de cada ${SLOW_FRAME_STRIDE} frames` : 'Video a tasa nativa de la cámara'}
        />
        <Stat
          icon={Database}
          label="Memoria GPU (O(1))"
          value={`${telemetry.vramMb.toFixed(1)} MB`}
          sub={`${telemetry.tensors} tensores activos`}
        />
        <Stat
          icon={Cpu}
          label="Backend activo"
          value={(backend || 'cargando').toUpperCase()}
          sub={[
            describeBackend(backendInfo),
            backendInfo?.benchmarks?.[backend] != null ? `${backendInfo.benchmarks[backend].toFixed(0)} ms/imagen al iniciar` : '',
          ]
            .filter(Boolean)
            .join(' · ')}
        />
        <div className="col-span-full flex items-center gap-2 rounded-xl bg-emerald-950 px-4 py-3 font-semibold text-emerald-300 ring-1 ring-emerald-700">
          <Wifi className="size-5 shrink-0" aria-hidden /> 100% Cliente (Sin Servidor Externo)
        </div>
      </aside>

      {error && (
        <div className="lg:col-span-2 flex flex-wrap items-center justify-between gap-3 rounded-xl bg-red-950 px-4 py-3 text-red-200 ring-1 ring-red-800">
          <span className="min-w-0 break-words">{error}</span>
          <button
            onClick={() => startCamera()}
            className="cursor-pointer rounded-lg bg-red-800 px-3 py-1.5 text-sm font-bold text-white hover:bg-red-700"
          >
            Reintentar cámara
          </button>
        </div>
      )}
    </div>
  )
}
