# ORDEN PARA CLAUDE CODE: EJECUTAR LA ETAPA 2

La Etapa 1 está COMPLETA. Antes de escribir código lee `docs/MODEL_CONTRACT.md` (valores confirmados con el modelo real) y corrige cualquier supuesto previo que lo contradiga.

## Hechos confirmados que cambian el prompt original
1. El modelo tiene **5 clases** (no 2), por este orden: `0 rider` (ignorar), `1 rider_full_face`, `2 rider_half_face`, `3 rider_helmet_invalid`, `4 rider_no_helmet`.
2. Salida del modelo: **`[1, 9, 8400]`**. Cajas en **píxeles 0-640** (no normalizadas).
3. Mapeo por NOMBRE leyendo `public/model/classes.json`:
   - CON casco: `rider_full_face`, `rider_half_face`
   - SIN casco: `rider_no_helmet`, `rider_helmet_invalid`
   - `rider` se descarta (y su canal 4 no entra al cálculo de confianza).
4. Archivos del modelo en `public/model/`: `model.json`, `group1-shard*of4.bin`, `classes.json`.

## Qué hacer ahora: ETAPA 2 (del prompt original)
- Scaffolding React + Vite + Tailwind CSS, dependencias `@tensorflow/tfjs` y `lucide-react`.
- `vite.config.js` y `vercel.json` para servir `/model/model.json` y los `.bin` con caché eficiente.
- Hook `useHelmetModel`: backend `webgl` con fallback a `cpu`, `tf.loadGraphModel('/model/model.json')`, warm-up con `tf.zeros([1,640,640,3])`, y estado (`loading`, `progress`, `backend`, `modelReady`, `error`). Debe cargar también `classes.json`.
- NO avances a la Etapa 3 hasta que `npm run build` pase sin errores. Avisa cuando termines.
