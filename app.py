import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
import time

# Define fruit quality classes
disease_types = [
    'Apple_Healthy',
    'Apple_Rotten',
    'Grape_Healthy',
    'Grape_Rotten',
    'Mango_Healthy',
    'Mango_Rotten',
    'Orange_Healthy',
    'Orange_Rotten'
]

def load_saved_model_weights(weights_path):
    input_shape = (128, 128, 3)
    inputs = Input(input_shape)
    efficient_net = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')(inputs)
    outputs = GlobalAveragePooling2D()(efficient_net)
    outputs = Dropout(0.25)(outputs)
    outputs = Dense(len(disease_types), activation='softmax')(outputs)
    model = Model(inputs, outputs)
    model.load_weights(weights_path)
    return model

def preprocess_uploaded_image(image_file, desired_size=128):
    if isinstance(image_file, Image.Image):
        img = image_file
    else:
        img = Image.open(image_file)
    img = img.resize((desired_size, desired_size))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_prediction_metrics(prediction):
    confidence = float(np.max(prediction))
    top_3_indices = np.argsort(prediction[0])[-3:][::-1]
    top_3_predictions = [(disease_types[i], float(prediction[0][i])) for i in top_3_indices]
    return confidence, top_3_predictions

st.set_page_config(
    page_title="Fruit Quality Detection",
    page_icon="🍎",
    layout="wide"
)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>
<style>
.main {
    background-color: #F8F9FA;
}
.stButton>button {
    background-color: #64A90C;
    color: white;
    padding: 0.8rem 1.5rem;
    border-radius: 0.5rem;
    border: none;
    transition: all 0.3s ease;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #50C878;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.prediction-box {
    padding: 1.8rem;
    border-radius: 1rem;
    background-color: #FFFFFF;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
    color: #212529;
    border: 1px solid #E9ECEF;
}
.result-text {
    font-size: 1.4rem;
    font-weight: bold;
}
h1 {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #64A90C 0%, #50C878 100%);
    color: white;
    border-radius: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(100,169,12,0.2);
}
.metric-card {
    background-color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
    color: #212529;
    transition: transform 0.3s ease;
    border: 1px solid #E9ECEF;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.healthy-text {
    color: #64A90C !important;
}
.rotten-text {
    color: #8B0000 !important;
}
.healthy-card {
    border-left: 4px solid #64A90C;
}
.rotten-card {
    border-left: 4px solid #8B0000;
}
.healthy-bg {
    background-color: #E8F5E9;
}
.rotten-bg {
    background-color: #FFEBEE;
}
.top-predictions {
    margin-top: 1.5rem;
    padding: 1.5rem;
    background-color: #F8F9FA;
    border-radius: 1rem;
    color: #212529;
}
.suggestion-card {
    background-color: #FFFFFF;
    padding: 1.2rem;
    border-radius: 0.8rem;
    margin-bottom: 0.8rem;
    color: #212529;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
.suggestion-card:hover {
    transform: translateX(5px);
}
div[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 0.8rem;
    border: 2px dashed #DEE2E6;
}
div[data-testid="stImage"] {
    background-color: #FFFFFF;
    padding: 1rem;
    border-radius: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.confidence-bar {
    padding: 0.8rem;
    border-radius: 0.5rem;
    margin-top: 0.8rem;
    font-weight: 500;
}
.footer {
    text-align: center;
    padding: 2rem;
    color: #6C757D;
    border-top: 1px solid #DEE2E6;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1><i class='fas fa-apple-alt'></i> Fruit Quality Detection System</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="prediction-box">
        <h3 class="healthy-text"><i class="fas fa-upload"></i> Upload Image</h3>
        <p style='color: #6C757D;'>Support formats: JPG, PNG, JPEG</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        with st.spinner('Processing image...'):
            start_time = time.time()
            
            image = Image.open(uploaded_file)
            resized_image = image.resize((224, 224))
            with col1:
                st.image(resized_image, caption="Uploaded Image")

            processed_image = preprocess_uploaded_image(image)
            model = load_saved_model_weights('Trained_Models/LastModel_weights.h5')
            prediction = model.predict(processed_image)
            confidence, top_3_predictions = get_prediction_metrics(prediction)
            processing_time = time.time() - start_time

            with col2:
                st.markdown("""
                <div class="prediction-box">
                    <h3 class="healthy-text"><i class="fas fa-chart-bar"></i> Analysis Results</h3>
                </div>
                """, unsafe_allow_html=True)

                is_healthy = "Healthy" in top_3_predictions[0][0]
                status_color = "healthy-text" if is_healthy else "rotten-text"
                bg_color = "healthy-bg" if is_healthy else "rotten-bg"
                icon = "leaf" if is_healthy else "skull"

                st.markdown(f"""
                <div class="prediction-box {bg_color}">
                    <p class="result-text {status_color}">
                        <i class="fas fa-{icon}"></i> Primary Detection
                    </p>
                    <h2 class="{status_color}">{top_3_predictions[0][0]}</h2>
                    <div class="confidence-bar {bg_color}">
                        <i class="fas fa-chart-line"></i> Confidence Level: {top_3_predictions[0][1]:.2%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                metrics_cols = st.columns(2)
                with metrics_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 class="{status_color}">
                            <i class="fas fa-percentage"></i> Confidence Score
                        </h4>
                        <div class="{status_color}" style="font-size: 1.5rem; font-weight: bold;">
                            {confidence:.2%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with metrics_cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 class="{status_color}">
                            <i class="fas fa-clock"></i> Processing Time
                        </h4>
                        <div class="{status_color}" style="font-size: 1.5rem; font-weight: bold;">
                            {processing_time:.2f}s
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="top-predictions">
                    <h4 class="healthy-text"><i class="fas fa-list-ol"></i> Detailed Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for disease, prob in top_3_predictions:
                    is_healthy_pred = "Healthy" in disease
                    card_class = "healthy-card" if is_healthy_pred else "rotten-card"
                    text_class = "healthy-text" if is_healthy_pred else "rotten-text"
                    pred_icon = "leaf" if is_healthy_pred else "skull"
                    
                    st.markdown(f"""
                    <div class="suggestion-card {card_class}">
                        <i class="fas fa-{pred_icon} {text_class}"></i> 
                        <b class="{text_class}">{disease}</b>
                        <div style='float: right; font-weight: 600;' class="{text_class}">
                            {prob:.2%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.markdown("""
<div class="footer">
    <p><i class="fas fa-code healthy-text"></i> Developed by </p>
    <p><i class="fas fa-brain healthy-text"></i> Model Architecture EfficientNetB0</p>
</div>
""", unsafe_allow_html=True)
