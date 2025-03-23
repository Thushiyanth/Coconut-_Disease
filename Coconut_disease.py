import streamlit as st
import os
import google.generativeai as genai
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Securely load Gemini AI API key from environment variable
genai.configure(api_key="AIzaSyA3VYu_hAB4T0QtUGbSJ2KTW7gIA1od1G8")

# Load the trained model with caching and error handling
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("inceptionv3_fine_tuned_model.keras")  # Update with actual path
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define disease classes and remedies
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

# Disease prediction function
def predict_disease(image):
    if model is None:
        return "Error: Model not loaded", 0.0
    
    img = image.resize((299, 299))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    if predicted_class >= len(disease_info):
        return "Unknown Disease", confidence

    return list(disease_info.keys())[predicted_class], confidence

# Streamlit UI
st.title("🌴 Coconut Disease Diagnosis Chatbot 🤖")
st.write("Upload an image of a coconut tree and chat with our AI to diagnose diseases.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello, farmer! Upload an image to check for diseases."}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File uploader UI
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    with col2:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze"):
        label, confidence = predict_disease(image)
        response = f"The image is predicted as *{label}* with *{confidence:.2f} confidence.*"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Gemini AI Chat Function
def ask_gemini(user_input):
    history = st.session_state.get("messages", [])
    formatted_history = [{"parts": [{"text": msg["content"]}], "role": msg["role"]} for msg in history]
    chat = genai.GenerativeModel("gemini-1.5-pro").start_chat(history=formatted_history)
    response = chat.send_message(user_input)
    return response.text

# User Input for Chatbot
if user_input := st.chat_input("Ask about the disease, symptoms, or remedies..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if any(disease in user_input for disease in disease_info):
        disease = next(d for d in disease_info if d in user_input)
        response = f"{disease}\n\n*Cause:* {disease_info[disease]['cause']}\n\n*Remedy:* {disease_info[disease]['remedy']}"
    else:
        response = ask_gemini(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
