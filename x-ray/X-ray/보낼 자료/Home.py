import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import flask

# Load the pre-trained CNN model from your saved file
def load_cnn_model():
    model = tf.keras.models.load_model('C:/Users/myong/Desktop/x-ray/X-ray/Finish/훈련된 cnn 모델/my_trained_cnn_model.h5')  #본인 컴퓨터에 따라 경로 수정
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to 256x256
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit UI
st.set_page_config(page_title="MIAS", page_icon="🩻", initial_sidebar_state="expanded")
st.title("MIAS")
st.subheader("X-ray 사진을 업로드 해주세요.")

uploaded_image = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

    # Load and use the model for prediction
    model = load_cnn_model()
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    
    # Interpret the prediction
    result = "폐렴" if prediction[0] > 0.5 else "정상"
    st.subheader(f"결과: {result}")
