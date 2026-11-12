# Orden para Code — modelo 416 px (reemplaza al de 320)

El 320 perdía demasiado detalle del casco. Nuevo modelo entrenado a 416, exportado y verificado: `public/model_416/`.

## Contrato (verificado con imagen real, SavedModel)
- Entrada NHWC `[1,416,416,3]` float32 RGB 0..1 (letterbox gris 114).
- Salida `[1, 9, 3549]` (52²+26²+13² anclas; 4 cajas + 5 clases, sigmoide aplicada).
- Cajas en PIXELES 0–416 (`box_scale: "pixels_0_416"`).
- 4 shards (`group1-shard{1..4}of4.bin`).
- mAP50 val: all 0.575 (full_face 0.747, half_face 0.626, helmet_invalid 0.66, no_helmet 0.844). 640: ~0.56, 320: 0.546.

## Tareas
1. En `useHelmetModel.js`: `MODEL_URL='/model_416/model.json'`, `CLASSES_URL='/model_416/classes.json'`, subir `MODEL_VERSION` (la caché IndexedDB incluye imgsz, pero súbelo igual).
2. `public/model/` (hoy el 320) ya no se usa: no borrarlo todavía, solo dejar de referenciarlo. Respaldo del 640 en `public/model_640/`.
3. El motor ya lee imgsz/anclas/box_scale del modelo y de classes.json: confirmar que `boxScale` se pasa desde `classes.json` (`pixels_0_416` → píxeles) y que `scripts/test-engine.mjs` pasa con 416/3549.
4. Mantener pendientes de ORDEN_RENDIMIENTO.md (aviso/ fallback wasm si WebGL es por software, tiempos en consola).
