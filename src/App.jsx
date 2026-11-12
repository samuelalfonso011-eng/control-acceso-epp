import { useState, useCallback } from 'react'
import { Video, ImageIcon, Loader2, AlertTriangle, Bike } from 'lucide-react'
import { useHelmetModel } from './hooks/useHelmetModel'
import { evaluateAccess } from './utils/inferenceEngine.js'
import AccessBanner from './components/AccessBanner.jsx'
import LiveCamera from './components/LiveCamera.jsx'
import ImageAnalyzer from './components/ImageAnalyzer.jsx'

const TABS = [
  { id: 'live', label: 'Cámara en vivo', Icon: Video },
  { id: 'image', label: 'Analizar imagen', Icon: ImageIcon },
]

export default function App() {
  const { loading, progress, backend, backendInfo, error, model, classes } = useHelmetModel()
  const [tab, setTab] = useState('live')
  const [access, setAccess] = useState(() => evaluateAccess([]))

  const changeTab = useCallback((id) => {
    setTab(id)
    setAccess(evaluateAccess([]))
  }, [])

  return (
    <div className="mx-auto flex min-h-full max-w-7xl flex-col gap-5 p-4 sm:p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="flex min-w-0 items-center gap-3 text-xl font-black text-white sm:text-3xl">
          <Bike className="size-7 shrink-0 text-sky-400 sm:size-8" aria-hidden /> Control de Acceso Vehicular · EPP
        </h1>
        <nav className="flex w-full gap-2 sm:w-auto" role="tablist">
          {TABS.map(({ id, label, Icon }) => (
            <button
              key={id}
              role="tab"
              aria-selected={tab === id}
              onClick={() => changeTab(id)}
              className={`flex min-w-0 flex-1 cursor-pointer items-center justify-center gap-2 rounded-xl px-3 py-3 text-sm font-bold transition-colors sm:flex-none sm:px-5 sm:text-lg ${
                tab === id ? 'bg-sky-500 text-slate-950' : 'bg-slate-800 text-slate-200 hover:bg-slate-700'
              }`}
            >
              <Icon className="size-5 shrink-0" aria-hidden /> {label}
            </button>
          ))}
        </nav>
      </header>

      <AccessBanner access={access} />

      {loading && (
        <div className="flex flex-wrap items-center gap-3 rounded-2xl bg-slate-800/90 px-4 py-3 text-sm text-sky-300 ring-1 ring-slate-700 sm:px-5 sm:text-base">
          <Loader2 className="size-5 shrink-0 animate-spin" />
          <span className="min-w-0 flex-1 font-medium">
            Inicializando motor YOLOv8n en caché local… {Math.round(progress * 100)}%
          </span>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-700 sm:ml-auto sm:w-40">
            <div
              className="h-full bg-sky-400 transition-all duration-200"
              style={{ width: `${Math.round(progress * 100)}%` }}
            />
          </div>
        </div>
      )}

      {error && (
        <p className="flex items-center gap-3 rounded-2xl bg-red-950 px-4 py-4 text-base text-red-300 ring-1 ring-red-800 sm:px-5 sm:text-lg">
          <AlertTriangle className="size-6 shrink-0" /> <span className="min-w-0 break-words">{error}</span>
        </p>
      )}

      {tab === 'live' && (
        <LiveCamera model={model} classes={classes} backend={backend} backendInfo={backendInfo} onAccess={setAccess} />
      )}
      {tab === 'image' && <ImageAnalyzer model={model} classes={classes} backend={backend} onAccess={setAccess} />}
    </div>
  )
}
