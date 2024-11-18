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