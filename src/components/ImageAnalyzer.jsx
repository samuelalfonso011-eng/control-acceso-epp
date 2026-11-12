import { useCallback, useEffect, useRef, useState } from 'react'
import { ImageUp, Loader2, Sparkles } from 'lucide-react'
import { detect, drawDetections, evaluateAccess } from '../utils/inferenceEngine.js'
import { compressImageFile } from '../utils/imageCompressor.js'

const SAMPLES = [
  { file: 'moto-casco.jpg', label: '1 motociclista con casco' },
  { file: 'moto-copiloto-sin-casco.jpg', label: 'Conductor con casco, copiloto sin casco' },
  { file: 'moto-sin-casco.jpg', label: 'Ambos sin casco' },
]

function formatBytes(bytes) {
  if (!bytes) return '0 KB'
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export default function ImageAnalyzer({ model, classes, backend, onAccess }) {
  const imgRef = useRef(null)
  const canvasRef = useRef(null)
  const runId = useRef(0)
  const inFlightSrc = useRef(null)
  const objectUrl = useRef(null)
  const onAccessRef = useRef(onAccess)
  onAccessRef.current = onAccess

  const [src, setSrc] = useState(null)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)
  const [compressionStats, setCompressionStats] = useState(null)

  useEffect(
    () => () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
      onAccessRef.current(evaluateAccess([]))
    },
    [],
  )

  // Invalida cualquier análisis en curso y deja el banner en espera (talanquera abajo):
  // el resultado de la imagen ANTERIOR nunca debe mostrarse sobre una imagen nueva.
  const resetForNewImage = () => {
    runId.current++
    inFlightSrc.current = null
    const c = canvasRef.current
    c?.getContext('2d').clearRect(0, 0, c.width, c.height)
    onAccessRef.current(evaluateAccess([]))
  }

  const analyze = useCallback(async () => {
    const img = imgRef.current
    if (!img || !img.naturalWidth || !model || !classes) return
    // onLoad y el efecto de "modelo listo" pueden dispararse para la misma imagen: una sola inferencia
    if (inFlightSrc.current === img.src) return
    inFlightSrc.current = img.src
    const id = ++runId.current
    setBusy(true)
    setError(null)
    try {
      const { detections, result, latencyMs } = await detect(model, classes, img)
      if (id !== runId.current) return
      drawDetections(canvasRef.current, detections, img.naturalWidth, img.naturalHeight)
      onAccessRef.current(result)
      setCompressionStats((prev) => (prev ? { ...prev, latencyMs } : { latencyMs, width: img.naturalWidth, height: img.naturalHeight }))
    } catch (e) {
      console.error(e)
      setError(e.message)
    } finally {
      if (id === runId.current) {
        inFlightSrc.current = null
        setBusy(false)
      }
    }
  }, [model, classes])

  // Si el usuario eligió una imagen antes de que el modelo terminara de cargar, analiza apenas esté listo
  useEffect(() => {
    if (model && classes && src && imgRef.current?.complete && imgRef.current?.naturalWidth) {
      analyze()
    }
  }, [model, classes, src, analyze])

  const show = (url, stats = null) => {
    if (objectUrl.current && objectUrl.current !== url) {
      URL.revokeObjectURL(objectUrl.current)
      objectUrl.current = null
    }
    resetForNewImage()
    setCompressionStats(stats)
    if (url === src) {
      // Misma imagen otra vez: onLoad no se dispara, así que se re-analiza directamente
      analyze()
    } else {
      setBusy(true)
      setSrc(url)
    }
  }

  const loadFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      return setError('El archivo seleccionado no es una imagen válida.')
    }
    resetForNewImage()
    setBusy(true)
    setError(null)
    try {
      // Compresión en código con buffer O(1) antes de renderizar e inferir
      const compressed = await compressImageFile(file, { maxDim: 960, quality: 0.8 })
      objectUrl.current = compressed.url
      const savedPct =
        compressed.originalBytes > 0
          ? Math.max(0, Math.round((1 - compressed.compressedBytes / compressed.originalBytes) * 100))
          : 0
      show(compressed.url, {
        originalBytes: compressed.originalBytes,
        compressedBytes: compressed.compressedBytes,
        savedPct,
        width: compressed.width,
        height: compressed.height,
        latencyMs: null,
      })
    } catch (e) {
      console.error(e)
      setError(`No se pudo comprimir/cargar la imagen: ${e.message}`)
      setBusy(false)
    }
  }

  return (
    <div className="grid grid-cols-1 gap-4">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          loadFile(e.dataTransfer.files[0])
        }}
        className={`relative grid min-h-64 min-w-0 place-items-center sm:min-h-80 overflow-hidden rounded-2xl border-2 border-dashed bg-slate-900 p-3 transition-colors ${
          dragging ? 'border-sky-400 bg-slate-800' : 'border-slate-600'
        }`}
      >
        {src ? (
          <div className="relative inline-block max-w-full">
            <img
              ref={imgRef}
              src={src}
              alt="Imagen analizada"
              onLoad={analyze}
              className="block max-h-[60vh] max-w-full rounded-lg"
            />
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full" />
          </div>
        ) : (
          <label className="flex cursor-pointer flex-col items-center gap-3 text-center text-slate-300">
            <ImageUp className="size-12 sm:size-14" aria-hidden />
            <span className="text-lg font-bold sm:text-xl">Arrastra o toca para elegir una imagen</span>
            <span className="text-slate-400">Se comprimirá automáticamente en memoria (O(1)) antes de analizar</span>
            <input type="file" accept="image/*" className="sr-only" onChange={(e) => loadFile(e.target.files[0])} />
          </label>
        )}
        {busy && (
          <div className="absolute right-3 top-3 flex items-center gap-2 rounded-xl bg-slate-950/85 px-3 py-2 text-sky-300">
            <Loader2 className="size-5 animate-spin" /> Procesando…
          </div>
        )}
      </div>

      {compressionStats && (
        <div className="flex flex-wrap items-center gap-3 rounded-xl bg-slate-900 px-4 py-2.5 text-sm text-slate-300 ring-1 ring-slate-800">
          <Sparkles className="size-4 text-emerald-400" aria-hidden />
          {compressionStats.originalBytes ? (
            <span>
              <strong>Compresión en código:</strong> {formatBytes(compressionStats.originalBytes)} →{' '}
              {formatBytes(compressionStats.compressedBytes)} ({compressionStats.width}×{compressionStats.height}px
              {compressionStats.savedPct > 0 ? `, -${compressionStats.savedPct}% RAM` : ''})
            </span>
          ) : (
            <span>
              <strong>Dimensiones:</strong> {compressionStats.width}×{compressionStats.height}px
            </span>
          )}
          {compressionStats.latencyMs != null && (
            <span className="font-mono text-emerald-400 sm:ml-auto">
              Inferencia{backend ? ` (${backend.toUpperCase()})` : ''}: {compressionStats.latencyMs.toFixed(0)} ms
            </span>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <label className="w-full cursor-pointer rounded-xl bg-sky-500 px-5 py-3 text-center font-bold text-slate-950 hover:bg-sky-400 sm:w-auto">
          Subir imagen
          <input type="file" accept="image/*" className="sr-only" onChange={(e) => loadFile(e.target.files[0])} />
        </label>
        <span className="w-full text-sm font-semibold uppercase tracking-wide text-slate-400 sm:w-auto">Galería de prueba:</span>
        {SAMPLES.map((s) => (
          <button
            key={s.file}
            onClick={() => show(`/samples/${s.file}`)}
            className="w-full cursor-pointer rounded-xl bg-slate-800 px-4 py-3 text-left sm:w-auto font-semibold text-slate-100 ring-1 ring-slate-600 hover:bg-slate-700"
          >
            {s.label}
          </button>
        ))}
      </div>

      {error && <p className="rounded-xl bg-red-950 px-4 py-3 break-words text-red-300 ring-1 ring-red-800">{error}</p>}
    </div>
  )
}
