import streamlit as st
import os
import google.generativeai as genai
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import gdown

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="🌴 Coconut Disease Diagnosis Bot", layout="centered")
st.title("🌴 Coconut Disease Diagnosis Chatbot 🤖")
st.write("Upload an image of a coconut tree or leaf and chat with our AI to diagnose diseases.")

# ------------------ GEMINI API KEY ------------------
genai.configure(api_key="AIzaSyA3VYu_hAB4T0QtUGbSJ2KTW7gIA1od1G8")  # Replace with your actual API Key

# ------------------ MODEL FILE SETUP ------------------
TREE_MODEL_PATH = "tree_model.keras"
LEAF_MODEL_PATH = "leaf_model.keras"

TREE_MODEL_ID = "1Qse74IbkhvuMCVytroGzvpT-9E6DuEU9"  # Replace with your tree model ID
LEAF_MODEL_ID = "1gUT8FKVCisPaFRl8efdavLmPzNx5fXLU"  # Replace with your leaf model ID

TREE_MODEL_URL = f"https://drive.google.com/uc?id={TREE_MODEL_ID}"
LEAF_MODEL_URL = f"https://drive.google.com/uc?id={LEAF_MODEL_ID}"

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_tree_model():
    if not os.path.exists(TREE_MODEL_PATH):
        st.info("Downloading tree model from Google Drive...")
        gdown.download(TREE_MODEL_URL, TREE_MODEL_PATH, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(TREE_MODEL_PATH)

@st.cache_resource
def load_leaf_model():
    if not os.path.exists(LEAF_MODEL_PATH):
        st.info("Downloading leaf model from Google Drive...")
        gdown.download(LEAF_MODEL_URL, LEAF_MODEL_PATH, quiet=False, fuzzy=True)
    return tf.keras.models.load_model(LEAF_MODEL_PATH)

tree_model = load_tree_model()
leaf_model = load_leaf_model()

# ------------------ DISEASE INFO ------------------
disease_info = {
    "BudRootDropping": {
        "cause": "Caused by fungal infection due to excess moisture.",
        "remedy": "Use fungicides and ensure proper drainage."
    },
    "BudRot": {
        "cause": "Caused by Phytophthora fungus affecting young palms.",
        "remedy": "Apply Bordeaux mixture and prune affected parts."
    },
    "LeafRot": {
        "cause": "Occurs due to fungal attack in humid conditions.",
        "remedy": "Use copper-based fungicides and remove infected leaves."
    },
    "StemBleeding": {
        "cause": "Caused by a fungal infection leading to dark gum exudation.",
        "remedy": "Scrape infected areas and apply fungicidal paste."
    }
}

# ------------------ PREDICTION FUNCTION ------------------
def predict_disease(image, model):
    img = image.resize((299, 299))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    if predicted_class >= len(disease_info):
        return "Unknown Disease", confidence

    return list(disease_info.keys())[predicted_class], confidence

# ------------------ IMAGE UPLOAD SECTION ------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    image_type = st.radio(
        "Please choose the type of image / பட வகையைத் தேர்ந்தெடுக்கவும்:",
        ("🌴 Tree / மரம்", "🍃 Leaf / இலை")
    )

    selected_model = tree_model if "Tree" in image_type else leaf_model

    if st.button("🔍 Analyze Disease"):
        label, confidence = predict_disease(image, selected_model)
        response = f"✅ Predicted disease: *{label}*\n\n🎯 Confidence: *{confidence:.2f}*"
        if label in disease_info:
            response += f"\n\n🧪 *Cause:* {disease_info[label]['cause']}\n💊 *Remedy:* {disease_info[label]['remedy']}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# ------------------ CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello, farmer! Upload an image and ask about coconut diseases!"
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ GEMINI AI CHATBOT ------------------
def ask_gemini(user_input):
    model = genai.GenerativeModel(
        "gemini-1.5-pro",
        system_instruction=(
            "You are a helpful assistant that only answers questions related to coconut diseases, "
            "their symptoms, causes, remedies, and coconut farming. If asked anything else, reply with: "
            "'I'm sorry, I can only help with coconut-related queries.' "
            "You must understand Tamil queries and respond in Tamil language. If the user expects a reply in Tamil, "
            "give the reply in Tamil. Also, if the user input is in Tamil, understand it and reply in Tamil."
        )
    )

    history = [
        {"role": msg["role"], "parts": [msg["content"]]}
        for msg in st.session_state.get("messages", [])
    ]

    chat = model.start_chat(history=history)
    response = chat.send_message(user_input)
    return response.text

# ------------------ USER TEXT CHAT INPUT ------------------
if user_input := st.chat_input("Ask about coconut diseases or remedies..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    matched_disease = next((d for d in disease_info if d.lower() in user_input.lower()), None)

    if matched_disease:
        response = (
            f"🦠 *{matched_disease}*\n\n"
            f"🧪 *Cause:* {disease_info[matched_disease]['cause']}\n"
            f"💊 *Remedy:* {disease_info[matched_disease]['remedy']}"
        )
    else:
        response = ask_gemini(user_input)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
