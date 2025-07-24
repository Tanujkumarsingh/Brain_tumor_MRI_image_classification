import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# App title
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")

# Class labels
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Load model safely
MODEL_PATH = "brain_tumor_mobilenetv2.h5"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please ensure 'brain_tumor_mobilenetv2.h5' is in the app directory.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Upload file
file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])
if file is not None:
    with st.spinner("🔍 Classifying..."):
        # Display image
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction)
        pred_class = class_names[pred_index]
        confidence = np.max(prediction) * 100

        # Display prediction
        st.success(f"🎯 **Prediction:** `{pred_class}`")
        st.info(f"📊 **Confidence:** `{confidence:.2f}%`")
