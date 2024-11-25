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
model_path = "D:/minki_project/x-ray/X-ray/Finish/훈련된 cnn 모델/my_trained_cnn_model.h5"  # 실제 모델 경로를 확인해주세요.
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
    uploaded_img_pred_class = np.argmax(uploaded_img_pred)

    # 5. 결과 출력
		st.image(uploaded_img, caption="업로드된 X-ray 이미지", width=true)  # 이미지 크기 조정
		st.subheader(f"예측 결과: {'정상' if uploaded_img_pred_class == 0 else '비정상'}")
    
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가
st.write("")  # 세로 공간 추가


# 도넛 차트 그리기 함수
def draw_donut_chart(labels, sizes, colors):    #(라벨,사이즈,색)
    plt.style.use('dark_background')    #차트 배경색 (검정)
    fig, ax = plt.subplots(figsize=(6, 6))  # 차트 크기 조정
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=140)

    # 중앙에 흰색 원을 추가하여 도넛 모양 만들기
    centre_circle = plt.Circle((0, 0), 0.70, fc='Black')
    fig.gca().add_artist(centre_circle)
    
    # 비율 텍스트 설정
    plt.setp(autotexts, size=15, weight="bold", color="white")
    ax.set_title("Circle Chart")

    return fig

# 선형 차트 그리기 함수
def draw_line_chart(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_title("Line Chart")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid()
    return fig

def show():
    st.title("결과 및 측정내용")

    # 3개의 열 생성 ( Column 3섹션으로 나누기 )
    col1, spacer1, col2, spacer2, col3 = st.columns([30, 5, 40, 5, 30])


    # Column 1: 도넛 차트
    with col1:
        st.header("Column 1: Circle Chart")
        st.write("This column is Kategorie.")
        # 데이터 준비
        # 한글 폰트 설정 (윈도우에서 "Malgun Gothic" 사용)
        rc('font', family='Malgun Gothic')

        # 한글 깨짐 방지 설정 (특히 마이너스 기호 깨짐 방지)
        plt.rcParams['axes.unicode_minus'] = False
        labels = ["원외폐렴", "원내폐렴", "역사회획득폐렴", "시중폐렴"]
        sizes = [17, 27, 42, 1]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # 도넛 차트 그리기
        donut_chart = draw_donut_chart(labels, sizes, colors)
        st.pyplot(donut_chart)

    # 선형 차트 생성 함수
    def draw_line_chart(x, y):
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='o', color='blue', label="선형 그래프")
        ax.set_title("Custom X and Y Line Chart")
        ax.set_xlabel("X-Side")
        ax.set_ylabel("Y-Side")
        ax.legend()
        return fig

    # Column 2: 선형 차트
        col2 = st.columns(1)[0]  # 한 열을 사용
    with col2:
        st.header("Column 2: Line Chart")
        st.write("This chart shows the relationship between X and Y values.")
    
        # X 값: [0, 5, 10, 15, 20, 25, 30]
        x = np.array([0, 5, 10, 15, 20, 25, 30])
    
        # Y 값: 0.7에서 1.0 사이의 임의의 값 생성
        y = np.random.uniform(0.7, 1.0, size=len(x))
    
        # 선형 차트 그리기
        line_chart = draw_line_chart(x, y)
        st.pyplot(line_chart)
        
    # Column 3: 메트릭
    with col3:
        st.header("Column 3: cumulative patients")
        st.metric(label="Total patients", value="759474", delta="100")
        st.metric(label="New patients", value="150", delta="20")
        st.metric(label="death rate", value="5%", delta="-1%")

        with st.expander('About', expanded=True):
             st.write('''
            - Data: [U.S. Census Bureau](<https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html>).
            - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
            - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
            ''')

if __name__ == "__main__":
    show()

