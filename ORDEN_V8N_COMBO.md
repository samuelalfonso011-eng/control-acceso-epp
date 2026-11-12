# Orden para Code — modelo final: YOLOv8n 416 entrenado con los 2 datasets

Modelo nuevo en `public/model_v8n_combo/` (4 shards, ~12 MB). Reemplaza al v8s (demasiado lento: 74–82 s en CPU).
- Entrada `[1,416,416,3]` float32 RGB 0..1, letterbox gris 114.
- Salida `[1, 9, 3549]`, cajas en PIXELES 0–416 (`box_scale: "pixels_0_416"`).
- Mismas 5 clases y orden (0 rider ignorar, 1 full_face, 2 half_face, 3 helmet_invalid, 4 no_helmet).
- Validación (514 imágenes): mAP50 full_face 0.74, half_face 0.66, helmet_invalid 0.62, no_helmet 0.87.

## Prueba hecha (motor real de la app, umbral asimétrico casco ≥ 0.50 / sin casco ≥ 0.30)
| Imagen | Esperado | v8n 416 | v8s | v8n combo |
|---|---|---|---|---|
| casco abierto blanco | Verde | Rojo | Verde 0.50 | **Verde 0.64** |
| moto-casco.jpg | Verde | Rojo | Verde 0.51 | **Verde 0.82** |
| copiloto sin casco | Rojo | Rojo | Rojo (0.37) | **Rojo (0.79)** |
| ambos sin casco (x2 fotos) | Rojo | Rojo | Rojo | **Rojo** |
| 3 personas con casco | Verde | Verde | Verde | **Verde** |

## Tareas
1. `useHelmetModel.js`: `MODEL_URL='/model_v8n_combo/model.json'`, `CLASSES_URL='/model_v8n_combo/classes.json'`; clave de caché con arquitectura + dataset (`helmet-yolov8n-416-combo-vN`).
2. Mantener el umbral asimétrico (casco 0.50, sin casco 0.30).
3. Sacar `model_v8s` del `public/` publicado (ya está en `model_backups`), para no subir 42 MB a Vercel.
4. `npm run build` y `node scripts/test-engine.mjs` deben pasar; probar las 3 muestras en el navegador y reportar ms por imagen.
