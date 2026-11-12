import { ShieldCheck, ShieldAlert, Hourglass, Users, HardHat, UserX } from 'lucide-react'
import { ACCESS } from '../utils/inferenceEngine.js'

const THEME = {
  [ACCESS.ALLOWED]: { bg: 'bg-emerald-600', ring: 'ring-emerald-300', Icon: ShieldCheck, arm: '#34d399', raised: true },
  [ACCESS.DENIED]: { bg: 'bg-red-600', ring: 'ring-red-300', Icon: ShieldAlert, arm: '#f87171', raised: false },
  [ACCESS.IDLE]: { bg: 'bg-slate-700', ring: 'ring-sky-300', Icon: Hourglass, arm: '#94a3b8', raised: false },
}

/** Talanquera: el brazo rota sobre su pivote; levantada solo en ALLOWED. */
function Barrier({ color, raised }) {
  return (
    <svg viewBox="0 0 220 120" className="w-40 max-w-full shrink-0 sm:w-56" role="img" aria-label={raised ? 'Talanquera levantada' : 'Talanquera bloqueada'}>
      <rect x="10" y="70" width="34" height="46" rx="4" fill="#e2e8f0" />
      <rect x="10" y="70" width="34" height="10" rx="3" fill="#475569" />
      <g
        style={{
          transformOrigin: '40px 72px',
          transform: raised ? 'rotate(-70deg)' : 'rotate(0deg)',
          transition: 'transform 600ms cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      >
        <rect x="38" y="64" width="170" height="14" rx="4" fill={color} />
        <rect x="80" y="64" width="16" height="14" fill="#fff" opacity="0.85" />
        <rect x="116" y="64" width="16" height="14" fill="#fff" opacity="0.85" />
        <rect x="152" y="64" width="16" height="14" fill="#fff" opacity="0.85" />
      </g>
      <circle cx="40" cy="72" r="7" fill="#1e293b" />
      <rect x="196" y="72" width="8" height="44" rx="2" fill="#64748b" />
    </svg>
  )
}

function Counter({ icon: Icon, label, value }) {
  return (
    <div className="flex min-w-0 items-center gap-3 rounded-2xl bg-black/25 px-4 py-3 sm:px-5">
      <Icon className="size-7 shrink-0 sm:size-8" aria-hidden />
      <div className="min-w-0">
        <div className="text-3xl font-black tabular-nums leading-none sm:text-4xl">{value}</div>
        <div className="text-xs font-semibold uppercase tracking-wide opacity-90 sm:text-sm">{label}</div>
      </div>
    </div>
  )
}

export default function AccessBanner({ access }) {
  const t = THEME[access.status]
  const Icon = t.Icon
  return (
    <section
      aria-live="polite"
      className={`${t.bg} ring-4 ${t.ring} rounded-3xl px-4 py-5 text-white shadow-2xl transition-colors duration-300 sm:px-6 sm:py-6`}
    >
      <div className="flex flex-col items-center gap-4 sm:gap-6 lg:flex-row lg:justify-between">
        <div className="flex min-w-0 max-w-full items-center gap-3 sm:gap-5">
          <Icon className="size-10 shrink-0 sm:size-16 lg:size-24" aria-hidden />
          <h2 className="min-w-0 text-2xl font-black leading-tight break-words sm:text-4xl lg:text-6xl">{access.message}</h2>
        </div>
        <Barrier color={t.arm} raised={t.raised} />
      </div>
      <div className="mt-5 grid grid-cols-1 gap-3 min-[480px]:grid-cols-3 lg:flex lg:flex-wrap lg:justify-start">
        <Counter icon={Users} label="Ocupantes detectados" value={access.total} />
        <Counter icon={HardHat} label="Con casco" value={access.withHelmetCount} />
        <Counter icon={UserX} label="Sin casco" value={access.withoutHelmetCount} />
      </div>
    </section>
  )
}
