# %% [markdown]
# # Etapa 1 - Entrenamiento YOLOv8n (Casco / Sin Casco) y exportación a TensorFlow.js
# Runtime: **GPU (T4)**. Dataset: Roboflow `mapua-university/motorcycle-helmet-detection-dataset` **versión 8**.
# La API key va en *Secrets* de Colab (icono de llave) con el nombre `ROBOFLOW_API_KEY` y acceso al notebook activado. NUNCA pegues la key en el código.

# %%
# Celda 1 - Dependencias y verificación de GPU
!pip install -q ultralytics roboflow tensorflowjs
import torch
print("GPU disponible:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(usa CPU: muy lento, cambia a T4)")

# %%
# Celda 2 - Descarga del dataset (formato yolov8)
from google.colab import userdata
from roboflow import Roboflow

rf = Roboflow(api_key=userdata.get("ROBOFLOW_API_KEY"))
project = rf.workspace("mapua-university").project("motorcycle-helmet-detection-dataset")
dataset = project.version(8).download("yolov8")
DATA_YAML = f"{dataset.location}/data.yaml"
print("data.yaml:", DATA_YAML)

# %%
# Celda 3 - FUENTE DE VERDAD de las clases (copiar este resultado al contrato)
import yaml, json

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

names = cfg["names"]
if isinstance(names, list):
    names = {i: n for i, n in enumerate(names)}
names = {int(k): v for k, v in names.items()}
print("nc =", len(names))
print("CLASES (orden exacto del data.yaml):")
for i, n in sorted(names.items()):
    print(f"  {i}: {n}")

# %%
# Celda 4 - Entrenamiento
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data=DATA_YAML,
    imgsz=640,
    epochs=50,        # sube a 80-100 si mAP50 < 0.85
    batch=16,
    patience=15,
    device=0,
    project="runs",
    name="helmet_v8n",
    exist_ok=True,
)

# %%
# Celda 5 - Validación rápida
best_pt = "runs/helmet_v8n/weights/best.pt"
metrics = YOLO(best_pt).val(data=DATA_YAML, imgsz=640)
print("mAP50:", metrics.box.map50, "| mAP50-95:", metrics.box.map)

# %%
# Celda 6 - Exportación a TensorFlow.js (GraphModel)
# Genera runs/helmet_v8n/weights/best_web_model/ con model.json + group1-shard*of*.bin
exported = YOLO(best_pt).export(format="tfjs", imgsz=640)
print("Exportado en:", exported)

# %%
# Celda 7 - Verificación del contrato con una IMAGEN REAL del dataset
import glob, os, numpy as np, cv2, tensorflow as tf

web_dir = "runs/helmet_v8n/weights/best_web_model"
files = sorted(os.listdir(web_dir))
print(files)
assert "model.json" in files and any(f.endswith(".bin") for f in files), "FALLO: el export tfjs no generó model.json/.bin"

saved = glob.glob("runs/helmet_v8n/weights/best_saved_model")
assert saved, "FALLO: no existe best_saved_model, no se puede verificar la escala de cajas"

img_path = sorted(glob.glob(f"{dataset.location}/valid/images/*"))[0]
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
x = (cv2.resize(img, (640, 640)).astype("float32") / 255.0)[None]

fn = tf.saved_model.load(saved[0]).signatures["serving_default"]
inp = list(fn.structured_input_signature[1].keys())[0]
t = list(fn(**{inp: tf.constant(x)}).values())[0].numpy()
print("Forma de salida:", t.shape, "-> esperado (1, %d, 8400)" % (4 + len(names)))
assert t.shape == (1, 4 + len(names), 8400), "FALLO: forma de salida distinta al contrato"

best = t[0, 4:].max(axis=0).argmax()
bx = t[0, :4, best]
print("Mejor ancla -> cx,cy,w,h =", bx, "| score =", t[0, 4:, best].max())
box_scale = "normalized_0_1" if t[0, :4].max() <= 1.5 else "pixels_0_640"
print("ESCALA DE CAJAS:", box_scale)

# %%
# Celda 8 - Guardar classes.json (incluye box_scale) y empaquetar para descargar
meta = {
    "imgsz": 640,
    "input": "NHWC [1,640,640,3], float32, RGB, 0..1",
    "output": f"[1, {4 + len(names)}, 8400]",
    "box_scale": box_scale,
    "classes": {str(k): v for k, v in sorted(names.items())},
}
with open(f"{web_dir}/classes.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(json.dumps(meta, indent=2, ensure_ascii=False))

!cd runs/helmet_v8n/weights && zip -r /content/best_web_model.zip best_web_model

from google.colab import files
files.download("/content/best_web_model.zip")
