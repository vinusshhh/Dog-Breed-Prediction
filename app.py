# ===== IMPORTS (MUST BE AT THE TOP) =====
import streamlit as st
import tensorflow as tf
import json
import numpy as np
import requests
from PIL import Image

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Dog Breed Predictor 🐶", layout="centered")

st.title("🐕 Dog Breed Prediction App")
st.write("Upload a dog image and the model will predict its breed.")

# ===== DOG API =====
API_KEY = "live_1jxGwydpE97RBZgS8clefjOCFVSWqyMCdO7kTOsrytH8NpXDoIm3LZfUb2MD2qKY"

def get_breed_info(breed_name):

    url = "https://api.thedogapi.com/v1/breeds/search"

    headers = {
        "x-api-key": API_KEY
    }

    response = requests.get(
        url,
        headers=headers,
        params={"q": breed_name}
    )

    if response.status_code == 200:
        data = response.json()

        if len(data) > 0:
            return data[0]

    return None

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

    breed_info = get_breed_info(clean_breed)

    # ===== DISPLAY RESULT =====
    st.markdown("## 🏆 Predicted Dog Breed")
    st.success(f"**{clean_breed}**")

    st.markdown("### 🔎 Confidence")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}%")

    if breed_info:

        st.markdown("---")

        st.markdown(f"""
        ### 🐶 About {breed_info.get('name', clean_breed)}

        **⚖️ Weight:** {breed_info.get('weight', {}).get('metric', 'N/A')} kg

        **⏳ Life Span:** {breed_info.get('life_span', 'N/A')}

        **😊 Temperament:** {breed_info.get('temperament', 'N/A')}

        **🏷️ Breed Group:** {breed_info.get('breed_group', 'N/A')}
        """)

    # ===== RESET BUTTON =====
    if st.button("🔄 Reset"):
        st.session_state.reset = True
        st.rerun()

# ===== CLEAR STATE AFTER RESET =====
if st.session_state.reset:
    st.session_state.reset = False
