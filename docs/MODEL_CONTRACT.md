# Contrato del modelo YOLOv8n -> TensorFlow.js

Contrato entre el entrenamiento (Colab) y el frontend. **Modelo vigente: detección por CABEZAS**
(`public/model_heads/`, YOLOv8n a 416 px, dataset `cabezas_3_datasets`: 3 datasets de Roboflow Universe).
Reemplaza al modelo anterior de ocupantes completos (`rider_*`, 5 clases), que veía "moto + persona" como
un solo bulto y fallaba con 2+ ocupantes o con personas sin moto. Ver `ORDEN_CABEZAS.md`.

## 1. Diccionario de clases (orden exacto de classes.json)

| ID | Nombre      | Rol de negocio                  | Color   |
|----|-------------|---------------------------------|---------|
| 0  | `con_casco` | CON casco: suma a withHelmetCount    | #10B981 |
| 1  | `sin_casco` | SIN casco: suma a withoutHelmetCount | #EF4444 |

**Mapear SIEMPRE por nombre** (el listado vive en `public/model_heads/classes.json`), no por ID fijo.
El mapeo está en `src/utils/classMap.js`:

```js
export const HELMET_OK  = ['con_casco']
export const HELMET_BAD = ['sin_casco']
```

- Cada caja es una **cabeza**: una caja = un ocupante. Conductor y copiloto salen como cajas separadas.
- Ya no existen la clase genérica `rider` ni "casco inválido"; una clase que no esté en ninguna de las
  dos listas se ignora.
- El modelo cuenta **cualquier cabeza visible**, también peatones cerca de la moto (un peatón sin casco
  deniega el acceso). Mitigación pendiente: zona de control (ROI) en la cámara en vivo.

## 2. Tensores

**Entrada**
- Forma: `[1, 416, 416, 3]` (NHWC), `float32`, RGB, valores `0..1` (píxel / 255).
- Letterbox a 416x416 con relleno gris (114); guardar `scale` y `padX/padY` para revertir cajas.
- El tamaño lo dicta `imgsz` en classes.json; el hook comprueba que coincida con `model.json`.

**Salida** (único tensor)
- Forma: **`[1, 6, 3549]`** (4 caja + 2 clases).
- 3549 = 52x52 + 26x26 + 13x13 anclas. El motor lee el número real de `output.shape[2]`.
- Canales: `0..3` = `x_center, y_center, width, height`; `4..5` = score por clase (ya con sigmoide).
- No hay "objectness" separado: confianza = score de la clase.
- **Escala de cajas: PÍXELES del espacio 416x416** (`box_scale: "pixels_0_416"`). NO multiplicar por 416.

**Post-procesado** (`src/utils/inferenceEngine.js`)
- Recorrido lineal del buffer plano, sin transponer.
- Umbral asimétrico (ante la duda se niega): `helmet >= 0.50`, `noHelmet >= 0.30`. Se configura en el
  campo `thresholds` de classes.json.
- NMS agnóstico a la clase, `iou = 0.45`, máximo 20 detecciones; luego des-letterbox.

## 3. Archivos en el proyecto web

```
public/
  model_heads/
    model.json
    group1-shard1of4.bin ... group1-shard4of4.bin   (~12.6 MB en total)
    classes.json            <- imgsz, box_scale, thresholds y mapeo de clases
```

- Para actualizar el modelo (p. ej. tras re-entrenar): copiar **todo** el contenido de `best_web_model/`
  (sin la carpeta contenedora ni `best.pt`) a `public/model_heads/` y **subir `MODEL_VERSION`** en
  `src/hooks/useHelmetModel.js`; si no, el navegador sigue usando la copia vieja guardada en IndexedDB.
- Los `.bin` conservan su nombre: `model.json` los referencia por ruta relativa.
- Carga en el cliente: `tf.loadGraphModel('/model_heads/model.json')`.
- Los pesos PyTorch (`.pt`) y los `.zip` de exportación van en `model_backups/`, **no** en `public/`:
  todo lo que hay en `public/` se publica con la web.

## 4. Verificación

- `node scripts/test-engine.mjs`: post-procesado con tensores sintéticos (6 canales, 2 clases).
- `scripts/letterbox.ps1` + `node scripts/diag-fotos.mjs <carpeta>`: decisión de la app y confianzas del
  modelo real sobre fotos, sin navegador.

## 5. Nota de reproducibilidad (exportación)

En Colab con Python 3.13 + Ultralytics 8.4, `export(format="tfjs")` NO genera TF.js (produce `.tflite`).
Se exportó en un entorno aislado Python 3.11 (`ultralytics==8.3.40`, `tensorflow==2.15.1`, `tensorflowjs==4.17.0`,
`torch==2.4.1`, `onnx==1.16.2`, `onnx2tf==1.26.3`) y luego `tensorflowjs_converter --input_format=tf_saved_model
--output_format=tfjs_graph_model` sobre `best_saved_model`.
