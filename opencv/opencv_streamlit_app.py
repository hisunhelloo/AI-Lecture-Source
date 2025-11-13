"""
OpenCV 이미지 처리 Streamlit 앱 (모듈화 버전)
"""
import streamlit as st
from pages_basic import page_image_upload, page_image_transform, page_draw_shapes
from pages_advanced import (
    page_color_space,
    page_normalization,
    page_noise_filtering,
    page_fourier_transform,
    page_image_pyramid,
    page_edge_detection
)

# 페이지 설정
st.set_page_config(page_title="OpenCV Image Processing", layout="wide")

st.title("🖼️ OpenCV 이미지 처리 앱")

# 사이드바에서 기능 선택
page = st.sidebar.selectbox(
    "기능 선택",
    ["이미지 업로드 및 보기", "이미지 변환", "도형 그리기", "색 공간 변환", "정규화 및 표준화",
     "노이즈 및 필터링", "푸리에 변환", "이미지 피라미드", "경계 검출"]
)

# 페이지 라우팅
if page == "이미지 업로드 및 보기":
    page_image_upload()
elif page == "이미지 변환":
    page_image_transform()
elif page == "도형 그리기":
    page_draw_shapes()
elif page == "색 공간 변환":
    page_color_space()
elif page == "정규화 및 표준화":
    page_normalization()
elif page == "노이즈 및 필터링":
    page_noise_filtering()
elif page == "푸리에 변환":
    page_fourier_transform()
elif page == "이미지 피라미드":
    page_image_pyramid()
elif page == "경계 검출":
    page_edge_detection()

# 사이드바 정보 표시
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 OpenCV 이미지 처리")
st.sidebar.markdown("""
이 앱은 OpenCV를 사용한 이미지 처리 기능을 제공합니다:

**1장: OpenCV 기초**
- 이미지 읽기 및 표시
- 이미지 변환 (크기, 대칭, 회전, 자르기)
- 도형 그리기 (선, 원, 사각형, 타원, 다각형, 격자)

**2장: 이미지 처리 기초**
- 색 공간 변환 (RGB, Grayscale, HSV)
- 정규화 및 표준화
- 노이즈 생성 및 필터링
- 푸리에 변환 및 주파수 필터링
- 이미지 피라미드
- 경계 검출 (Canny, Sobel, Prewitt)
""")
