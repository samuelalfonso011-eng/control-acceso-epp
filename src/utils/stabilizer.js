import { evaluateAccess } from './inferenceEngine.js'

/**
 * Estabilizador de estado (debounce por consenso).
 * Un estado nuevo solo se adopta si se mantiene ininterrumpido `holdMs` (~350 ms);
 * un frame ruidoso aislado no hace parpadear el banner.
 */
export function createStabilizer(holdMs = 350) {
  const initial = evaluateAccess([])
  let stable = initial
  let candidate = null
  let since = 0

  return {
    push(result, now = performance.now()) {
      if (result.status === stable.status) {
        stable = result // mismo estado: refresca contadores
        candidate = null
        return stable
      }
      if (!candidate || candidate.status !== result.status) {
        candidate = result
        since = now
      } else {
        candidate = result
      }
      if (now - since >= holdMs) {
        stable = result
        candidate = null
      }
      return stable
    },
    reset() {
      stable = initial
      candidate = null
      return stable
    },
  }
}
