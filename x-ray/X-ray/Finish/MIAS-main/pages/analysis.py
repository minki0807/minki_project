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
