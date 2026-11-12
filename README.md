# Control de Acceso Vehicular · EPP

Aplicación web que verifica si los ocupantes de una motocicleta llevan casco y decide si se
permite el acceso. La detección corre **por completo en el navegador** con TensorFlow.js: no hay
servidor, base de datos ni envío de imágenes a terceros.

## Qué hace

- **Cámara en vivo:** analiza el video de la cámara y muestra en tiempo real si el acceso está
  permitido o denegado, con el conteo de ocupantes con y sin casco.
- **Analizar imagen:** permite subir una foto (o usar la galería de prueba) y obtener el mismo
  resultado sobre una imagen fija.
- **Regla de acceso:** se permite el paso solo si se detecta al menos un ocupante y todos llevan
  casco. Basta un ocupante sin casco para denegar.

## Cómo funciona

- Modelo **YOLOv8n** entrenado para detectar cabezas, con dos clases: `con_casco` y `sin_casco`.
  Cada caja detectada es un ocupante.
- Entrada de 416×416 px. El modelo se exporta a TensorFlow.js y vive en `public/model_heads/`.
- Al iniciar, la app mide los backends disponibles (WebGL y WASM) y usa el más rápido.
- El modelo se guarda en IndexedDB tras la primera carga, así las siguientes visitas arrancan
  más rápido.

El detalle de tensores, clases y umbrales está en [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md).

## Requisitos

- Node.js 20 o superior
- Un navegador moderno (Chrome, Edge, Firefox o Safari recientes)
- Para la cámara, la página debe abrirse por HTTPS (o en `localhost` durante el desarrollo)

## Uso local

```bash
npm install
npm run dev      # servidor de desarrollo
npm run build    # compilación de producción en dist/
npm run preview  # sirve la compilación de producción
```

No se necesita ninguna variable de entorno (ver [.env.example](.env.example)).

## Despliegue

El proyecto está preparado para Vercel: basta importar el repositorio y desplegar con la
configuración detectada (Vite). [vercel.json](vercel.json) define las cabeceras necesarias:

- `Cross-Origin-Opener-Policy` y `Cross-Origin-Embedder-Policy`, que habilitan el modo multihilo
  del backend WASM.
- Cabeceras de seguridad y de caché para el modelo y los recursos estáticos.

## Estructura

```
src/
  components/   Banner de acceso, cámara en vivo y analizador de imágenes
  hooks/        Carga del modelo y selección de backend
  utils/        Inferencia, mapeo de clases, estabilizador y compresión de imágenes
public/
  model_heads/  Modelo en formato TensorFlow.js
  samples/      Imágenes de la galería de prueba
colab/          Notebook y script de entrenamiento (YOLOv8n) y exportación
docs/           Contrato del modelo
scripts/        Utilidades de diagnóstico y pruebas del motor de inferencia
```

## Entrenamiento

El modelo se entrena en Google Colab con el notebook de [colab/](colab/). La clave de Roboflow se
lee desde los *Secrets* de Colab con el nombre `ROBOFLOW_API_KEY`; nunca se escribe en el código.
