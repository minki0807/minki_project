import streamlit as st
from PIL import Image
import numpy as np

# 제목
st.title("MIAS")

# 사진 업로드
uploaded_file1 = st.file_uploader("첫 번째 사진 업로드", type=["jpg", "png"])
uploaded_file2 = st.file_uploader("두 번째 사진 업로드", type=["jpg", "png"])

if uploaded_file1 and uploaded_file2:
    # 이미지 열기
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)

    # 이미지 출력
    st.image(image1, caption='첫 번째 사진', use_column_width=True)
    st.image(image2, caption='두 번째 사진', use_column_width=True)

    # 이미지 비교 (크기 확인 및 차이 계산)
    img1_array = np.array(image1)
    img2_array = np.array(image2)

    if img1_array.shape == img2_array.shape:
        # 차이 계산
        difference = np.abs(img1_array - img2_array)
        st.image(difference, caption='차이 이미지', use_column_width=True)
        
        # 차이 계산 결과 출력
        st.write("두 이미지의 차이를 계산했습니다.")
    else:
        st.warning("사진의 크기가 다릅니다. 같은 크기의 이미지를 업로드해 주세요.")