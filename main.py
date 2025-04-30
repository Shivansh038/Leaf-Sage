import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set page config
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f9f4;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2e7d32;
            text-align: center;
            margin-bottom: 20px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #888;
            margin-top: 50px;
        }
        .button {
            background-color: #66bb6a !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🌿 Plant Disease Classifier 🌿</div>', unsafe_allow_html=True)

# Load model and classes
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Image preprocessing
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Prediction
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# File uploader
uploaded_image = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=250)

    with col2:
        st.markdown("### 🧪 Ready to Diagnose?")
        if st.button('🔍 Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'🌱 **Prediction:** {prediction}')


