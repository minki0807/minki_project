# --- Content from /mnt/data/X-ray CNN 최종.py ---
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# 경로 설정
main_path = "C:/Users/myong/Desktop/x-ray/input/archive/chest_xray/chest_xray"
train_path = os.path.join(main_path, "train")
test_path = os.path.join(main_path, "test")
val_path = os.path.join(main_path, "val")

# 데이터 증강 설정 (강화된 데이터 증강)
train_datagen = ImageDataGenerator(
    rescale=1./255
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 로딩
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(256, 256), #이미지 크기
    batch_size=64, # 배치 사이즈 조절
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(256, 256), #이미지 크기
    batch_size=64, # 배치 사이즈 조절
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(256, 256), #이미지 크기
    batch_size=64, # 배치 사이즈 조절
    class_mode='binary'
)

# CNN 모델 구성
model = Sequential()

# Convolution Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))  # input_shape 수정
model.add(BatchNormalization())  # Batch Normalization 추가
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution Layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution Layer 4
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # 드롭아웃 유지
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy',
              optimizer= 'Adam',  # 기본 학습률
              metrics=['accuracy'])

# 콜백 설정 (EarlyStopping + ReduceLROnPlateau)
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)

# 모델 학습
history = model.fit(
    train_generator,
    epochs=200,
    validation_data=val_generator,
    callbacks=[es]
)

# 학습 결과 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label="Training accuracy")
plt.plot(val_acc, label="Validation accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.5, 1)
plt.legend()
plt.title("Training vs Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.legend()
plt.title("Training vs Validation Loss")

plt.tight_layout()
plt.show()

# 모델 평가
test_loss, test_acc = model.evaluate(test_generator, steps=3000)
print('Test accuracy:', test_acc)


# --- Content from /mnt/data/analysis.py ---
import streamlit as st


# 페이지 제목 설정
st.set_page_config(
    page_title="MISA",
    page_icon="🩻",
)


import streamlit as st
from PIL import Image

def show():
    st.title("Image Analysis & Comparison")

    # 두 이미지 업로드
    uploaded_image1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    uploaded_image2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

    if uploaded_image1 and uploaded_image2:
        # 이미지 열기
        image1 = Image.open(uploaded_image1)
        image2 = Image.open(uploaded_image2)

        # 이미지 크기 비교
        size1 = image1.size  # (width, height)
        size2 = image2.size  # (width, height)

        # 이미지 출력
        # 2개의 열 생성
        col1, col2 = st.columns(2)

        with col1:
            st.image(image1, caption="Image 1", use_column_width=True)

        with col2:
            st.image(image2, caption="Image 2", use_column_width=True)

        # 크기 비교 결과 출력
        if size1 == size2:
            st.success(f"The images are of the same size: {size1}.")
        else:
            st.error(f"The images have different sizes: Image 1 size: {size1}, Image 2 size: {size2}.")

if __name__ == "__main__":
    show()


# --- Content from /mnt/data/results.py ---
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px



# 페이지 제목 설정
st.set_page_config(
    page_title="MISA",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded")
    


alt.themes.enable("dark")

# 도넛 차트 그리기 함수
def draw_donut_chart(labels, sizes, colors):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 6))  # 차트 크기 조정
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=140)

    # 중앙에 흰색 원을 추가하여 도넛 모양 만들기
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    
    # 비율 텍스트 설정
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.set_title("Donut Chart Example")

    return fig

# 선형 차트 그리기 함수
def draw_line_chart(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_title("Line Chart Example")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid()
    return fig

def show():
    st.title("차트 및 측정내용")

    # 3개의 열 생성
    col1, spacer1, col2, spacer2, col3 = st.columns([30, 5, 40, 5, 30])


    # Column 1: 도넛 차트
    with col1:
        st.header("Column 1: Donut Chart")
        st.write("This column is narrow.")
        # 데이터 준비
        labels = ['Category A', 'Category B', 'Category C', 'Category D']
        sizes = [15, 30, 45, 10]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # 도넛 차트 그리기
        donut_chart = draw_donut_chart(labels, sizes, colors)
        st.pyplot(donut_chart)

    # Column 2: 선형 차트
    with col2:
        st.header("Column 2: Line Chart")
        st.write("This column is wider.")
        # 임의의 데이터 생성
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        # 선형 차트 그리기
        line_chart = draw_line_chart(x, y)
        st.pyplot(line_chart)

    # Column 3: 메트릭
    with col3:
        st.header("Column 3: Metrics")
        st.write("This column is narrow.")
        st.metric(label="Total Sales", value="$10,000", delta="$1,000")
        st.metric(label="New Customers", value="150", delta="20")
        st.metric(label="Conversion Rate", value="5%", delta="-1%")

        with st.expander('About', expanded=True):
             st.write('''
            - Data: [U.S. Census Bureau](<https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html>).
            - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
            - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
            ''')

if __name__ == "__main__":
    show()

# --- Content from /mnt/data/Home.py ---
import streamlit as st
from PIL import Image
import numpy as np


# 페이지 제목 설정
st.set_page_config(
    page_title="MISA",
    page_icon="🩻",
    initial_sidebar_state="expanded")



# 제목
st.title("MIAS")
# 소제목
st.subheader('안녕하세요. 이거 만드는데 겁나 힘들었어요.. 이 홈엔 ')
st.subheader('팀원 소개라던지 간단한 문구 넣으면 어떨까요??')


