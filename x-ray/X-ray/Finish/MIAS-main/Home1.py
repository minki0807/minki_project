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

    # 결과 출력
    st.image(uploaded_img, caption="업로드된 X-ray 이미지", use_column_width=True)
    st.write(f"예측 결과: {'폐렴' if uploaded_img_pred_class == 1 else '정상'}")

# 도넛 차트 그리기 함수
def draw_donut_chart(labels, sizes, colors):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=140)
    centre_circle = plt.Circle((0, 0), 0.70, fc='black')
    fig.gca().add_artist(centre_circle)
    plt.setp(autotexts, size=15, weight="bold", color="white")
    ax.set_title("Circle Chart")
    return fig

# 선형 차트 그리기 함수
def draw_line_chart(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', color='blue', label="선형 그래프")
    ax.set_title("Custom X and Y Line Chart")
    ax.set_xlabel("X-Side")
    ax.set_ylabel("Y-Side")
    ax.legend()
    return fig

def show():
    st.title("결과 및 측정 내용")

    # 3개의 열 생성
    col1, spacer1, col2, spacer2, col3 = st.columns([30, 5, 40, 5, 30])

    # Column 1: 도넛 차트
    with col1:
        st.header("Column 1: Circle Chart")
        st.write("This column is Kategorie.")

        rc('font', family='Malgun Gothic')
        plt.rcParams['axes.unicode_minus'] = False

        labels = ["원외폐렴", "원내폐렴", "역사회획득폐렴", "시중폐렴"]
        sizes = [17, 27, 42, 14]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

        donut_chart = draw_donut_chart(labels, sizes, colors)
        st.pyplot(donut_chart)

    # Column 2: 선형 차트
    with col2:
        st.header("Column 2: Line Chart")
        st.write("This chart shows the relationship between X and Y values.")

        x = np.array([0, 5, 10, 15, 20, 25, 30])
        y = np.random.uniform(0.5, 1.0, size=len(x))

        line_chart = draw_line_chart(x, y)
        st.pyplot(line_chart)

    # Column 3: 메트릭
    with col3:
        st.header("Column 3: Cumulative Patients")
        st.metric(label="Total Patients", value="759,474", delta="100")
        st.metric(label="New Patients", value="150", delta="20")
        st.metric(label="Death Rate", value="5%", delta="-1%")

        with st.expander('About', expanded=True):
            st.write('''
                - Data: [U.S. Census Bureau](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html).
                - **Gains/Losses**: States with high inbound/outbound migration for the selected year.
                - **States Migration**: Percentage of states with annual inbound/outbound migration > 50,000.
            ''')

if __name__ == "__main__":
    show()
