# Orden para Code — modelo nuevo por CABEZAS (reemplaza al v8n combinado)

## Por qué
El modelo anterior marcaba "moto + persona" como una sola caja: con 2+ ocupantes veía un solo bulto y no reconocía a una persona sin moto (cámara en vivo). El nuevo detecta CABEZAS, una caja por ocupante.

## Modelo: `public/model_heads/` (4 shards, ~12 MB) — YOLOv8n 416
- Entrada `[1,416,416,3]` float32 RGB 0..1, letterbox gris 114 (igual que antes).
- Salida `[1, 6, 3549]` (4 caja + **2 clases**), cajas en píxeles 0–416 (`box_scale: "pixels_0_416"`).
- Clases: `0 con_casco`, `1 sin_casco`. Ya no existen `rider_*` ni "casco inválido".
- `classes.json` trae `thresholds: { helmet: 0.5, noHelmet: 0.3 }`.
- Validación: mAP50 ≈ 0.75 (26 épocas, 3 datasets de Universe, 4.596 imágenes de entrenamiento).
- Pesos PyTorch de respaldo: `model_backups/heads_best.pt`.

## Prueba hecha con el motor actual (solo cambiando classMap), umbral 0.5 / 0.3
| Imagen | Esperado | Resultado |
|---|---|---|
| p3 (1 sin casco) | Rojo | Rojo |
| p4 (moto llena sin cascos) | Rojo | Rojo, 9 cabezas |
| p6, p7, p8 (2 con casco) | Verde | Verde, cuenta 2 |
| p9 (ella con, él sin) | Rojo | Rojo, 1 + 1 |
| prueba2 (4 en una moto) | Rojo | Rojo, 4 cabezas |
| 3 con casco (toboga...) | Verde | Verde, cuenta 3 |
| 3 muestras de public/samples | V / R / R | V / R / R |
| p5 (2 con casco + policía de pie sin casco) | Verde | Rojo: cuenta al peatón |

## Tareas
1. `src/utils/classMap.js`: `HELMET_OK = ['con_casco']`, `HELMET_BAD = ['sin_casco']` (el motor ya mapea por nombre; no hay clase a ignorar).
2. `inferenceEngine.js`: `LABELS = { con_casco: 'Casco', sin_casco: 'Sin Casco' }`. Quitar comentarios/ramas de `rider_helmet_invalid`.
3. `useHelmetModel.js`: apuntar a `/model_heads/model.json` y `/model_heads/classes.json` (o copiar el contenido a `public/model/`), y cambiar la clave de caché (`helmet-heads-yolov8n-416-vN`) para no cargar el modelo viejo de IndexedDB.
4. Las cajas ahora son pequeñas (cabezas): revisar grosor de línea y tamaño de etiqueta en `drawDetections`.
5. Peatones: el modelo cuenta cualquier cabeza visible. Opcional y recomendado para la cámara en vivo: una zona de control (ROI central configurable) y contar solo cabezas cuyo centro caiga dentro.
6. Actualizar `docs/MODEL_CONTRACT.md` y `scripts/test-engine.mjs` (6 canales, 2 clases). `npm run build` debe pasar.
7. Probar en el navegador las fotos p3–p9 y prueba2 y la cámara en vivo, y reportar resultado y ms.
