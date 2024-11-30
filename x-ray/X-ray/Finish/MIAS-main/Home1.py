# home.py
import streamlit as st
from streamlit import runtime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # 맵차트 허용
import pandas as pd   #라이브러리로부터 Series나 DataFrame을 불러오는
import altair as alt    # 바 차트 허용
import plotly.express as px     # 그래프 차트 허용
from skimage.metrics import structural_similarity as ssim
import cv2
from matplotlib import rc
import tensorflow as tf


#-------straemlit 부분

# 제목
st.title("MIAS")
# 소제목
st.subheader('폐렴 X-Ray.')

st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가

# 모델 로드 (올바른 경로로 지정)
model_path = 'C:/Users/myong/Desktop/x-ray/X-ray/Finish/훈련된 cnn 모델/my_trained_cnn_model.h5'  # 실제 모델 경로를 확인해주세요.
model = tf.keras.models.load_model(model_path)  # TensorFlow 2.x에서 올바른 함수 사용

# X-ray 이미지 업로드 (사용자가 업로드한 이미지)
uploaded_file = st.file_uploader("업로드할 X-ray 이미지를 선택하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 3. 업로드된 이미지 열기
    uploaded_img = Image.open(uploaded_file)
    uploaded_img = uploaded_img.convert('L')  # 흑백 이미지로 변환 (X-ray는 흑백이므로)
    uploaded_img = uploaded_img.resize((150, 150))  # 모델 입력 크기에 맞게 크기 조정
    uploaded_img_array = np.array(uploaded_img) / 255.0  # 정규화
    uploaded_img_array = np.expand_dims(uploaded_img_array, axis=-1)  # 배치 차원 추가

    # 4. 이미지 예측 수행
    uploaded_img_pred = model.predict(np.expand_dims(uploaded_img_array, axis=0))

    # 예측 결과 클래스 (0: 정상, 1: 비정상)로 해석
    uploaded_img_pred_class = np.round(uploaded_img_pred[0])  # 예측 결과 반올림하여 0 또는 1로 변환

    # 예측 결과에 따른 출력
    st.image(uploaded_img, caption="업로드된 X-ray 이미지", width=300)  # 이미지 크기 조정

    if uploaded_img_pred_class == 0:
        st.success("예측 결과: 정상")
    else:
        st.error("예측 결과: 비정상")
        
