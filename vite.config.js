import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Aislamiento de origen: sin estas dos cabeceras no hay SharedArrayBuffer y el backend WASM
// corre con un solo hilo. Comprobar en consola: crossOriginIsolated === true.
// (Las mismas cabeceras están en vercel.json para producción.)
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

// Los tres binarios + el script UMD del backend. El UMD se sirve tal cual, sin pasar por el
// bundler: minificarlo rompe los workers del modo multihilo (ver src/utils/backend.js).
const WASM_FILES = [
  'tfjs-backend-wasm.wasm',
  'tfjs-backend-wasm-simd.wasm',
  'tfjs-backend-wasm-threaded-simd.wasm',
  'tf-backend-wasm.es2017.min.js',
]

/** Copia el backend WASM de TensorFlow.js a public/tfjs-wasm/ en cada `vite dev` / `vite build`. */
function copyTfjsWasm() {
  return {
    name: 'copy-tfjs-wasm',
    config() {
      const src = resolve('node_modules/@tensorflow/tfjs-backend-wasm/dist')
      const dst = resolve('public/tfjs-wasm')
      mkdirSync(dst, { recursive: true })
      for (const file of WASM_FILES) {
        const from = resolve(src, file)
        const to = resolve(dst, file)
        if (!existsSync(from)) throw new Error(`Falta ${from}. Ejecuta: npm install`)
        if (!existsSync(to) || statSync(to).size !== statSync(from).size) copyFileSync(from, to)
      }
    },
  }
}

export default defineConfig({
  plugins: [copyTfjsWasm(), react(), tailwindcss()],
  // Los .bin del modelo viven en public/model y se sirven tal cual (sin hash ni inlining).
  assetsInclude: ['**/*.bin'],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  build: {
    chunkSizeWarningLimit: 2500, // tfjs pesa ~1.5 MB minificado
  },
})
