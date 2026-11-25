import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.title("Clasificador de Residuos en la Nube - IA UTH 2026 - Madelin Borjas")

MODEL_DIR = Path("modelo_resiclaje_mobilenet")
MODEL_PATH_KERAS = MODEL_DIR / "waste_mobilenet.keras"
MODEL_PATH_H5 = MODEL_DIR / "waste_mobilenet.h5"
CLASSES_PATH = MODEL_DIR / "waste_mobilenet.json"
IMG_SIZE = (224, 224)

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "paper": "Papel",
    "plastic": "Plástico",
}


def load_class_names() -> list[str]:
    if not CLASSES_PATH.exists():
        st.error(f"No se encontró el archivo de clases: {CLASSES_PATH}")
        st.stop()
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    model_path = None
    if MODEL_PATH_KERAS.exists():
        model_path = MODEL_PATH_KERAS
    elif MODEL_PATH_H5.exists():
        model_path = MODEL_PATH_H5

    if model_path is None:
        st.error(
            "No se encontró el modelo. Sube al repositorio la carpeta "
            "'modelo_resiclaje_mobilenet' con 'waste_mobilenet.keras' "
            "o 'waste_mobilenet.h5' y 'waste_mobilenet.json'."
        )
        st.stop()

    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.stop()


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_all(model, class_names: list[str], image: Image.Image):
    x = prepare_image(image)
    preds = model.predict(x, verbose=0)[0]
    sorted_idx = np.argsort(preds)[::-1]
    results = []
    for idx in sorted_idx:
        raw_label = class_names[idx]
        show_label = LABELS_ES.get(raw_label, raw_label)
        results.append((raw_label, show_label, float(preds[idx]) * 100))
    return results, preds


st.write("Toma una fotografía desde la cámara o sube una imagen para clasificar el residuo.")
st.caption("Clases disponibles: vidrio, papel, cartón y plástico.")

class_names = load_class_names()
model = load_model()

tab_cam, tab_file = st.tabs(["📷 Usar cámara", "📁 Subir imagen"])

with tab_cam:
    camera_photo = st.camera_input("Toma una fotografía del residuo")

with tab_file:
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

input_source = camera_photo if camera_photo is not None else uploaded_file

if input_source is not None:
    image = Image.open(input_source)
    st.image(image, caption="Imagen capturada", use_container_width=True)

    results, preds = predict_all(model, class_names, image)
    principal = results[0]

    st.subheader("Predicción principal")
    st.success(f"Clase predicha: **{principal[1]}** — Confianza: **{principal[2]:.2f}%**")

    st.subheader("Detalle de probabilidades por clase")
    for i, raw_label in enumerate(class_names):
        label_es = LABELS_ES.get(raw_label, raw_label)
        st.progress(float(preds[i]), text=f"{label_es}: {preds[i]*100:.2f}%")
else:
    st.info("Toma una foto con la cámara o carga una imagen para comenzar.")
