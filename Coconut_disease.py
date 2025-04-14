import streamlit as st
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import gdown

# ------------------ CONFIG & SETUP ------------------
st.set_page_config(page_title="🦶 Diabetic Foot Ulcer Detector", layout="centered")
st.title("🦶 Diabetic Foot Ulcer Diagnosis System")
st.write("Upload an image of a foot and get instant prediction of ulcer condition.")

# ------------------ MODEL LOADING ------------------
MODEL_PATH = "dfu_inceptionv3_model.keras"
MODEL_ID = "1OeZ3DvUxI94lxuMYrY-8zOFIJMs6usY8"  # Update with your model's Drive ID
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------ DISEASE CLASS INFO ------------------
dfu_info = {
    "Diabetic Foot Ulcer": {
        "description": "Foot shows signs of ulceration, infection, or open wounds.",
        "recommendation": "Seek immediate medical attention. Keep the area clean and avoid pressure."
    },
    "Healthy Foot": {
        "description": "No visible signs of ulceration or abnormality.",
        "recommendation": "Maintain good foot hygiene and regularly monitor for signs of DFU."
    }
}

# ------------------ PREDICTION FUNCTION ------------------
def predict_dfu(image):
    image = image.resize((299, 299))  # Match InceptionV3 input size
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    predicted_class = "Diabetic Foot Ulcer" if prediction > 0.5 else "Healthy Foot"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return predicted_class, confidence

# ------------------ IMAGE UPLOAD SECTION ------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict DFU Condition"):
        label, confidence = predict_dfu(image)

        st.markdown(f"## ✅ Prediction: *{label}*")
        st.markdown(f"**🎯 Confidence:** `{confidence:.2f}`")

        st.markdown(f"**📌 Description:** {dfu_info[label]['description']}")
        st.markdown(f"**💡 Recommendation:** {dfu_info[label]['recommendation']}")
