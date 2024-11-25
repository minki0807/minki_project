# home.py
import streamlit as st
from streamlit import runtime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # 맵 차트 허용
import pandas as pd   # 라이브러리로부터 Series나 DataFrame을 불러오는
import altair as alt  # 바 차트 허용
import plotly.express as px  # 그래프 차트 허용
from skimage.metrics import structural_similarity as ssim
import cv2
from matplotlib import rc
import tensorflow as tf

#-------Streamlit 부분-------

# 제목
st.title("MIAS")
st.subheader('폐렴 X-Ray.')

# 모델 로드
model_path = 'C:/Users/myong/Desktop/x-ray/X-ray/Finish/훈련된 cnn 모델/my_trained_cnn_model.h5'  # 실제 모델 경로
model = tf.keras.models.load_model(model_path)

# X-ray 이미지 업로드
uploaded_file = st.file_uploader("업로드할 X-ray 이미지를 선택하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 업로드된 이미지 열기 및 전처리
    uploaded_img = Image.open(uploaded_file)
    uploaded_img = uploaded_img.convert('RGB')  # RGB로 변환
    uploaded_img = uploaded_img.resize((256, 256))  # 모델 입력 크기에 맞게 조정
    uploaded_img_array = np.array(uploaded_img) / 255.0  # 정규화
    uploaded_img_array = np.expand_dims(uploaded_img_array, axis=0)  # 배치 차원 추가

    # 이미지 예측 수행
    uploaded_img_pred = model.predict(uploaded_img_array)
    uploaded_img_pred_class = np.argmax(uploaded_img_pred)  # 클래스 (0: 정상, 1: 폐렴)

    st.title("결과 및 측정 내용")
    
    # 결과 출력
    st.image(uploaded_img, caption="업로드된 X-ray 이미지", use_column_width=True)
    st.subheader(f"예측 결과: {'폐렴' if uploaded_img_pred_class == 1 else '정상'}")