# Orden para Code — modelo 320 px y rendimiento

Modelo nuevo ya entrenado, exportado y verificado: `public/model_320/` (el 640 sigue en `public/model/` como respaldo).

## Contrato del modelo 320 (verificado con imagen real)
- Entrada NHWC `[1,320,320,3]` float32 RGB 0..1 (letterbox gris 114).
- Salida `[1, 9, 2100]` (4 cajas + 5 clases, sigmoide aplicada). 2100 = 40²+20²+10².
- Cajas en PIXELES 0–320 (`box_scale: "pixels_0_320"`).
- Clases por nombre: 0 rider (ignorar), 1 rider_full_face, 2 rider_half_face, 3 rider_helmet_invalid, 4 rider_no_helmet.
- mAP50 val: all 0.546 (full_face 0.732, half_face 0.614, helmet_invalid 0.537, no_helmet 0.847). El de 640 daba ~0.56.
- 3 shards (`group1-shard{1,2,3}of3.bin`), `classes.json` con imgsz 320.

## Tareas
1. `INPUT_SIZE = 320`; leer `imgsz` y `box_scale` de `classes.json` en vez de fijarlos. Deriva anclas/escala de ahí (pixels → dividir por imgsz al dibujar).
2. URL del modelo: `/model_320/model.json`. Versionar la clave de IndexedDB (`helmet-yolov8n-v2-320`) para no cargar el 640 cacheado.
3. Warm-up con `tf.zeros([1,320,320,3])` usando la misma vía que la inferencia real (`executeAsync` si es la que se usa).
4. Quitar `WEBGL_FORCE_F16_TEXTURES` y `WEBGL_DELETE_TEXTURE_THRESHOLD=0` (sin evidencia de ganancia, riesgo de precisión).
5. Detectar WebGL por software (`WEBGL_debug_renderer_info` con "Basic Render", "SwiftShader" o "llvmpipe"): caer a `wasm` (copiar los .wasm a `public/`) y mostrar aviso en pantalla.
6. Log de tiempos (preproceso / modelo / NMS) en consola.
7. Pendientes anteriores: bug de re-clic con la misma foto en ImageAnalyzer; reintento del bucle en LiveCamera tras error.
8. `npm run build` y `node scripts/test-engine.mjs` deben seguir pasando; ajusta los tests a 320/2100.
