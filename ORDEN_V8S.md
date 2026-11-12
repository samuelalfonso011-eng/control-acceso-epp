# Orden para Code — modelo YOLOv8s 416 (2 datasets)

Nuevo modelo en `public/model_v8s/` (11 shards, ~41 MB). Mismo contrato que el 416 nano:
- Entrada `[1,416,416,3]` float32 RGB 0..1, letterbox gris 114.
- Salida `[1, 9, 3549]`, cajas en PIXELES 0–416 (`box_scale: "pixels_0_416"`).
- Mismas 5 clases y mismo orden (0 rider ignorar, 1 full_face, 2 half_face, 3 helmet_invalid, 4 no_helmet).

Entrenado con Mapúa v8 + "Motorcycle Riders" (convertido a cajas de persona completa): 4.518 imágenes.

## Tareas
1. `useHelmetModel.js`: `MODEL_URL='/model_v8s/model.json'`, `CLASSES_URL='/model_v8s/classes.json'`, subir `MODEL_VERSION` y que la clave de IndexedDB incluya el nombre del modelo (`helmet-v8s-416-vN`), porque el imgsz es el mismo que el nano 416.
2. `SCORE_THRESHOLD`: bajar de 0.5 a 0.35 (este modelo da ~0.50 en cascos abiertos y ~0.37 en un copiloto sin casco; con 0.5 se pierden).
3. Mostrar barra de progreso real en la carga (41 MB): ya existe `onProgress`, solo verificar que se ve.
4. El modelo es ~3.5x más pesado que el nano (28 GFLOPs): mantener el aviso/fallback de WebGL por software de ORDEN_RENDIMIENTO.md y loguear ms por inferencia.
5. Mantener `public/model_416/` (nano) como alternativa liviana; no borrar nada.
6. `npm run build` y `node scripts/test-engine.mjs` deben pasar.
