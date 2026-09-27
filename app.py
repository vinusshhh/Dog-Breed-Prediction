import streamlit as st
import tensorflow as tf
import json
import numpy as np
import requests
from PIL import Image

#Page Configuration

st.set_page_config(page_title="Dog Breed Predictor 🐶", layout="centered")

st.title("🐕 Dog Breed Prediction App")
st.write("Upload a clear dog image and the model will predict its breed.")

#API

API_KEY = st.secrets["DOG_API_KEY"]

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

#Load Model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_breed_model.keras")

model = load_model()

#Load Labels

with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# Minimum confidence required (change if needed)
CONFIDENCE_THRESHOLD = 75.0

#Session State

if "reset" not in st.session_state:
    st.session_state.reset = False

#Image Upload

uploaded_file = st.file_uploader(
    "Upload a dog image",
    type=["jpg", "jpeg", "png"],
    key="uploader"
)

#Prediction

if uploaded_file is not None and not st.session_state.reset:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Reject blank / nearly plain images
    if np.std(np.array(image)) < 8:
        st.error("⚠️ The uploaded image appears to be blank or too plain.")
        st.info("Please upload a clear image containing a dog.")
        st.stop()

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index] * 100)

    breed = labels[class_index]
    clean_breed = breed.split("-")[1].replace("_", " ")

    # Reject low-confidence predictions
    if confidence < CONFIDENCE_THRESHOLD:
        st.error("⚠️ Couldn't recognize a dog clearly.")
        st.info(
            "Please upload a clear image of a single dog. "
            "Avoid blurry, blank, or unrelated images."
        )
        st.stop()

    # Get breed details
    breed_info = get_breed_info(clean_breed)

    #Results

    st.markdown("## 🏆 Predicted Dog Breed")
    st.success(f"**{clean_breed.title()}**")

    st.markdown("### 🔎 Confidence")
    st.progress(int(confidence))
    st.write(f"**{confidence:.2f}%**")

    if breed_info:
        st.markdown("---")
        st.markdown(f"### 🐶 About {breed_info.get('name', clean_breed.title())}")

        st.write(f"**⚖️ Weight:** {breed_info.get('weight', {}).get('metric', 'N/A')} kg")
        st.write(f"**⏳ Life Span:** {breed_info.get('life_span', 'N/A')}")
        st.write(f"**😊 Temperament:** {breed_info.get('temperament', 'N/A')}")
        st.write(f"**🏷️ Breed Group:** {breed_info.get('breed_group', 'N/A')}")

#Reset Button

if st.button("🔄 Reset"):
    st.session_state.reset = True
    st.rerun()

if st.session_state.reset:
    st.session_state.reset = False
