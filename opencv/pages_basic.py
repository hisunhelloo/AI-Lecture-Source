"""
OpenCV 기초 기능 페이지들 (1장)
- 이미지 업로드 및 보기
- 이미지 변환
- 도형 그리기
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import load_default_image, pil_to_numpy


def page_image_upload():
    """페이지 1: 이미지 업로드 및 보기"""
    st.header("이미지 업로드 및 보기")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("이미지 업로드")
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])

        use_default = st.checkbox("기본 이미지 사용 (like_lenna.png)", value=True)

        if use_default:
            image = load_default_image()
            if image is not None:
                st.session_state['current_image'] = image
            else:
                st.warning("기본 이미지를 찾을 수 없습니다. like_lenna.png 파일이 있는지 확인하세요.")

        elif uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            image = pil_to_numpy(pil_image)
            st.session_state['current_image'] = image

    with col2:
        if 'current_image' in st.session_state and st.session_state['current_image'] is not None:
            st.subheader("이미지 정보")
            img = st.session_state['current_image']
            st.write(f"**변수 타입:** {type(img)}")
            st.write(f"**이미지 배열의 형태:** {img.shape}")
            st.write(f"**데이터 타입:** {img.dtype}")

            st.image(img, caption="업로드된 이미지", use_container_width=True, clamp=True)


def page_image_transform():
    """페이지 2: 이미지 변환"""
    st.header("이미지 변환")

    if 'current_image' not in st.session_state or st.session_state['current_image'] is None:
        st.warning("먼저 '이미지 업로드 및 보기' 페이지에서 이미지를 업로드하세요.")
    else:
        image = st.session_state['current_image']

        transform_type = st.selectbox(
            "변환 방법 선택",
            ["크기 변환", "대칭 변환", "회전 변환", "이미지 자르기"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("원본 이미지")
            st.image(image, caption="원본", use_container_width=True, clamp=True)

        with col2:
            st.subheader("변환된 이미지")

            if transform_type == "크기 변환":
                resize_method = st.radio("크기 조정 방법", ["픽셀 지정", "배율 지정"])

                if resize_method == "픽셀 지정":
                    width = st.slider("너비", 50, 500, 200)
                    height = st.slider("높이", 50, 500, 200)
                    result = cv2.resize(image, (width, height))
                else:
                    scale = st.slider("배율", 0.5, 3.0, 1.5, 0.1)
                    result = cv2.resize(image, dsize=None, fx=scale, fy=scale)

                st.image(result, caption=f"크기 조정: {result.shape}", use_container_width=True, clamp=True)

            elif transform_type == "대칭 변환":
                flip_code = st.radio(
                    "대칭 방향",
                    [("상하 대칭", 0), ("좌우 대칭", 1), ("상하좌우 대칭", -1)],
                    format_func=lambda x: x[0]
                )
                result = cv2.flip(image, flip_code[1])
                st.image(result, caption="대칭 변환", use_container_width=True, clamp=True)

            elif transform_type == "회전 변환":
                angle = st.slider("회전 각도", -180, 180, 30)
                border_value = st.slider("배경 색상 (0=검정, 255=흰색)", 0, 255, 0)

                height, width = image.shape
                matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                result = cv2.warpAffine(image, matrix, (width, height), borderValue=border_value)
                st.image(result, caption=f"회전: {angle}도", use_container_width=True, clamp=True)

            elif transform_type == "이미지 자르기":
                st.write("자를 영역을 지정하세요:")
                height, width = image.shape

                col_a, col_b = st.columns(2)
                with col_a:
                    y1 = st.slider("시작 Y", 0, height-1, 50)
                    x1 = st.slider("시작 X", 0, width-1, 50)
                with col_b:
                    y2 = st.slider("끝 Y", y1+1, height, min(y1+100, height))
                    x2 = st.slider("끝 X", x1+1, width, min(x1+100, width))

                result = image[y1:y2, x1:x2].copy()
                st.image(result, caption=f"자른 영역: [{y1}:{y2}, {x1}:{x2}]", use_container_width=True, clamp=True)


def page_draw_shapes():
    """페이지 3: 도형 그리기"""
    st.header("도형 그리기")

    shape_type = st.selectbox(
        "도형 선택",
        ["선", "원", "사각형", "타원", "다각형", "격자"]
    )

    # 캔버스 설정
    canvas_width = st.slider("캔버스 너비", 300, 1500, 1000)
    canvas_height = st.slider("캔버스 높이", 200, 1000, 500)
    color = st.slider("도형 색상 (0=검정, 255=흰색)", 0, 255, 255)

    space = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    if shape_type == "선":
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("시작점 X", 0, canvas_width, 100)
            y1 = st.number_input("시작점 Y", 0, canvas_height, 100)
        with col2:
            x2 = st.number_input("끝점 X", 0, canvas_width, 800)
            y2 = st.number_input("끝점 Y", 0, canvas_height, 400)

        thickness = st.slider("선 두께", 1, 10, 3)
        space = cv2.line(space, (x1, y1), (x2, y2), color, thickness, 1)

    elif shape_type == "원":
        col1, col2 = st.columns(2)
        with col1:
            cx = st.number_input("중심 X", 0, canvas_width, canvas_width//2)
            cy = st.number_input("중심 Y", 0, canvas_height, canvas_height//2)
        with col2:
            radius = st.slider("반지름", 10, min(canvas_width, canvas_height)//2, 100)
            thickness = st.slider("선 두께", 1, 10, 4)

        space = cv2.circle(space, (cx, cy), radius, color, thickness, 1)

    elif shape_type == "사각형":
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("왼쪽 위 X", 0, canvas_width, 500)
            y1 = st.number_input("왼쪽 위 Y", 0, canvas_height, 200)
        with col2:
            x2 = st.number_input("오른쪽 아래 X", 0, canvas_width, 800)
            y2 = st.number_input("오른쪽 아래 Y", 0, canvas_height, 400)

        thickness = st.slider("선 두께", 1, 10, 5)
        space = cv2.rectangle(space, (x1, y1), (x2, y2), color, thickness, 1)

    elif shape_type == "타원":
        col1, col2 = st.columns(2)
        with col1:
            cx = st.number_input("중심 X", 0, canvas_width, 500)
            cy = st.number_input("중심 Y", 0, canvas_height, 300)
            axis_w = st.slider("가로 축 길이", 10, canvas_width//2, 300)
        with col2:
            axis_h = st.slider("세로 축 길이", 10, canvas_height//2, 200)
            angle = st.slider("회전 각도", 0, 360, 0)
            start_angle = st.slider("시작 각도", 0, 360, 90)
            end_angle = st.slider("끝 각도", 0, 360, 250)

        thickness = st.slider("선 두께", 1, 10, 4)
        space = cv2.ellipse(space, (cx, cy), (axis_w, axis_h), angle, start_angle, end_angle, color, thickness)

    elif shape_type == "다각형":
        st.write("다각형 그리기 (기본 예제)")

        obj1 = np.array([[300, 500], [500, 500], [400, 600], [200, 600]])
        obj2 = np.array([[600, 500], [800, 500], [700, 200]])

        draw_type = st.radio("그리기 유형", ["외곽선만", "채우기", "둘 다"])

        if draw_type in ["외곽선만", "둘 다"]:
            space = cv2.polylines(space, [obj1], True, color, 3)
        if draw_type in ["채우기", "둘 다"]:
            space = cv2.fillPoly(space, [obj2], color, 1)

    elif shape_type == "격자":
        grid_spacing = st.slider("격자 간격", 10, 100, 50)

        for x in range(0, space.shape[1], grid_spacing):
            cv2.line(space, (x, 0), (x, space.shape[0]), color, 1)

        for y in range(0, space.shape[0], grid_spacing):
            cv2.line(space, (0, y), (space.shape[1], y), color, 1)

    st.image(space, caption=f"{shape_type} 그리기 결과", use_container_width=True, clamp=True)
    st.write(f"캔버스 크기: {space.shape}")
