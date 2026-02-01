# ===== IMPORTS (MUST BE AT THE TOP) =====
import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Dog Breed Predictor 🐶", layout="centered")

st.title("🐕 Dog Breed Prediction App")
st.write("Upload a dog image and the model will predict its breed.")

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_breed_model.keras")

model = load_model()

# ===== LOAD LABELS =====
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# ===== SESSION STATE (RESET) =====
if "reset" not in st.session_state:
    st.session_state.reset = False

# ===== IMAGE UPLOAD =====
uploaded_file = st.file_uploader(
    "Upload a dog image",
    type=["jpg", "jpeg", "png"],
    key="uploader"
)

if uploaded_file is not None and not st.session_state.reset:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ===== PREPROCESS =====
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ===== PREDICT =====
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    breed = labels[class_index]
    confidence = prediction[class_index] * 100

    clean_breed = breed.split("-")[1].replace("_", " ")

    # ===== DISPLAY RESULT =====
    st.markdown("## 🏆 Predicted Dog Breed")
    st.success(f"**{clean_breed}**")

    st.markdown("### 🔎 Confidence")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}%")

    # ===== RESET BUTTON =====
    if st.button("🔄 Reset"):
        st.session_state.reset = True
        st.rerun()

# ===== CLEAR STATE AFTER RESET =====
if st.session_state.reset:
    st.session_state.reset = False
