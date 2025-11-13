"""
이미지 처리 기초 기능 페이지들 (2장)
- 색 공간 변환
- 정규화 및 표준화
- 노이즈 및 필터링
- 푸리에 변환
- 이미지 피라미드
- 경계 검출
"""
import streamlit as st
import cv2
import numpy as np
from utils import load_default_image, load_color_image, generate_salt_noise, generate_pepper_noise


def page_color_space():
    """페이지 4: 색 공간 변환"""
    st.header("색 공간 변환")

    # 컬러 이미지 로드
    if 'color_image' not in st.session_state:
        color_img = load_color_image()
        if color_img is not None:
            st.session_state['color_image'] = color_img

    if 'color_image' in st.session_state and st.session_state['color_image'] is not None:
        rgb_image = st.session_state['color_image']

        conversion_type = st.selectbox(
            "변환 방법 선택",
            ["RGB to Grayscale", "RGB to HSV"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("원본 RGB 이미지")
            st.image(rgb_image, caption="RGB", use_container_width=True)
            st.write(f"Shape: {rgb_image.shape}")

        with col2:
            st.subheader("변환된 이미지")

            if conversion_type == "RGB to Grayscale":
                # RGB를 BGR로 변환한 후 Grayscale로
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

                st.image(gray_image, caption="Grayscale", use_container_width=True, clamp=True)
                st.write(f"Shape: {gray_image.shape}")

                # 그레이스케일 변환 공식 설명
                st.markdown("**그레이스케일 변환 공식:**")
                st.latex(r"Y = 0.299 \times R + 0.587 \times G + 0.114 \times B")

                # 샘플 픽셀 값 계산
                if st.checkbox("샘플 픽셀 값 계산 보기"):
                    R, G, B = rgb_image[0, 0]
                    Y_calc = 0.299 * R + 0.587 * G + 0.114 * B
                    st.write(f"픽셀 [0,0] RGB 값: R={R}, G={G}, B={B}")
                    st.write(f"계산된 Y 값: {Y_calc:.2f}")
                    st.write(f"실제 Grayscale 값: {gray_image[0, 0]}")

            elif conversion_type == "RGB to HSV":
                # RGB를 HSV로 변환
                rgb_float = rgb_image.astype(np.float32) / 255.0
                hsv_image = cv2.cvtColor((rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

                # HSV 채널 분리
                h_channel = hsv_image[:, :, 0]
                s_channel = hsv_image[:, :, 1]
                v_channel = hsv_image[:, :, 2]

                # HSV 채널 표시
                col_h, col_s, col_v = st.columns(3)

                with col_h:
                    st.image(h_channel, caption="Hue Channel", use_container_width=True, clamp=True)
                with col_s:
                    st.image(s_channel, caption="Saturation Channel", use_container_width=True, clamp=True)
                with col_v:
                    st.image(v_channel, caption="Value Channel", use_container_width=True, clamp=True)

                st.write(f"HSV Shape: {hsv_image.shape}")
    else:
        st.warning("컬러 이미지를 로드할 수 없습니다.")


def page_normalization():
    """페이지 5: 정규화 및 표준화"""
    st.header("픽셀 값의 정규화와 표준화")

    if 'color_image' not in st.session_state:
        color_img = load_color_image()
        if color_img is not None:
            st.session_state['color_image'] = color_img

    if 'color_image' in st.session_state and st.session_state['color_image'] is not None:
        rgb_image = st.session_state['color_image'].astype(np.float32)

        method = st.selectbox("방법 선택", ["정규화 (Normalization)", "표준화 (Standardization)"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("원본 이미지")
            st.image(rgb_image.astype(np.uint8), caption="원본", use_container_width=True)
            st.write(f"샘플 픽셀 [0,0]: {rgb_image[0, 0]}")
            st.write(f"최소값: {rgb_image.min():.2f}, 최대값: {rgb_image.max():.2f}")

        with col2:
            st.subheader(f"{method} 결과")

            if method == "정규화 (Normalization)":
                normalized_image = rgb_image / 255.0
                st.image(normalized_image, caption="정규화된 이미지", use_container_width=True, clamp=True)
                st.write(f"샘플 픽셀 [0,0]: {normalized_image[0, 0]}")
                st.write(f"최소값: {normalized_image.min():.4f}, 최대값: {normalized_image.max():.4f}")

                st.markdown("**정규화 공식:**")
                st.latex(r"X_{normalized} = \frac{X}{255}")

            else:  # 표준화
                mean = np.mean(rgb_image)
                stddev = np.std(rgb_image)
                standardized_image = (rgb_image - mean) / stddev

                # 표준화된 이미지를 표시하기 위해 스케일 조정
                display_image = (standardized_image - standardized_image.min()) / (
                    standardized_image.max() - standardized_image.min()
                )
                st.image(display_image, caption="표준화된 이미지", use_container_width=True, clamp=True)
                st.write(f"샘플 픽셀 [0,0]: {standardized_image[0, 0]}")
                st.write(f"평균: {mean:.2f}, 표준편차: {stddev:.2f}")
                st.write(f"표준화 후 평균: {np.mean(standardized_image):.6f}")

                st.markdown("**표준화 공식:**")
                st.latex(r"X_{standardized} = \frac{X - \mu}{\sigma}")

    else:
        st.warning("이미지를 로드할 수 없습니다.")


def page_noise_filtering():
    """페이지 6: 노이즈 및 필터링"""
    st.header("노이즈 생성 및 필터링")

    if 'current_image' not in st.session_state or st.session_state['current_image'] is None:
        image = load_default_image()
        if image is not None:
            st.session_state['current_image'] = image

    if 'current_image' in st.session_state and st.session_state['current_image'] is not None:
        image = st.session_state['current_image']

        filter_type = st.selectbox(
            "필터 방법 선택",
            ["중앙값 필터링 (Salt & Pepper 노이즈)", "가우시안 필터링 (Gaussian 노이즈)"]
        )

        if filter_type == "중앙값 필터링 (Salt & Pepper 노이즈)":
            st.subheader("중앙값 필터링 - Salt & Pepper 노이즈 제거")

            noise_ratio = st.slider("노이즈 비율", 0.01, 0.2, 0.05, 0.01)
            kernel_size = st.slider("필터 커널 크기", 3, 11, 5, 2)

            # 노이즈 추가
            salted_image = generate_salt_noise(image, noise_ratio)
            peppered_image = generate_pepper_noise(salted_image, noise_ratio)

            # 중앙값 필터 적용
            filtered_image = cv2.medianBlur(peppered_image, kernel_size)

            # 결과 표시
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.image(image, caption="원본", use_container_width=True, clamp=True)
            with col2:
                st.image(salted_image, caption="Salt 노이즈", use_container_width=True, clamp=True)
            with col3:
                st.image(peppered_image, caption="Salt & Pepper", use_container_width=True, clamp=True)
            with col4:
                st.image(filtered_image, caption="중앙값 필터링", use_container_width=True, clamp=True)

        else:  # 가우시안 필터링
            st.subheader("가우시안 필터링 - Gaussian 노이즈 제거")

            noise_mean = st.slider("노이즈 평균", 0, 10, 0)
            noise_sigma = st.slider("노이즈 표준편차", 1, 50, 25)

            # 가우시안 노이즈 추가
            gaussian_noise = np.random.normal(noise_mean, noise_sigma, image.shape).astype('uint8')
            noisy_image = cv2.add(image, gaussian_noise)

            # 가우시안 블러 적용
            st.subheader("다양한 σ 값으로 필터링")
            sigma_values = [1, 5, 10]
            denoised_images = []

            for sigma in sigma_values:
                denoised = cv2.GaussianBlur(noisy_image, (0, 0), sigma)
                denoised_images.append(denoised)

            # 결과 표시
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.image(noisy_image, caption="노이즈 추가", use_container_width=True, clamp=True)
            with col2:
                st.image(denoised_images[0], caption=f"σ=1", use_container_width=True, clamp=True)
            with col3:
                st.image(denoised_images[1], caption=f"σ=5", use_container_width=True, clamp=True)
            with col4:
                st.image(denoised_images[2], caption=f"σ=10", use_container_width=True, clamp=True)

    else:
        st.warning("먼저 이미지를 업로드하세요.")


def page_fourier_transform():
    """페이지 7: 푸리에 변환"""
    st.header("푸리에 변환 및 주파수 필터링")

    if 'current_image' not in st.session_state or st.session_state['current_image'] is None:
        image = load_default_image()
        if image is not None:
            st.session_state['current_image'] = image

    if 'current_image' in st.session_state and st.session_state['current_image'] is not None:
        image = st.session_state['current_image']

        st.subheader("2D 푸리에 변환")

        # 푸리에 변환
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="원본 이미지", use_container_width=True, clamp=True)

        with col2:
            st.image(magnitude_spectrum, caption="주파수 스펙트럼", use_container_width=True, clamp=True)

        st.subheader("주파수 필터링")

        filter_type = st.radio("필터 유형", ["High-pass Filter (고주파 통과)", "Low-pass Filter (저주파 통과)"])
        radius = st.slider("필터 반경", 10, 100, 30)

        # 필터 적용
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        if filter_type == "High-pass Filter (고주파 통과)":
            # High-pass: 중앙(저주파)을 0으로, 나머지를 1로
            mask = np.ones((rows, cols), np.uint8)
            mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0
        else:
            # Low-pass: 중앙(저주파)을 1로, 나머지를 0으로
            mask = np.zeros((rows, cols), np.uint8)
            mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1

        fshift_filtered = fshift * mask

        # 역 푸리에 변환
        f_ishift = np.fft.ifftshift(fshift_filtered)
        image_back = np.fft.ifft2(f_ishift)
        image_back = np.abs(image_back)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(mask * 255, caption="필터 마스크", use_container_width=True, clamp=True)
        with col2:
            magnitude_filtered = 20 * np.log(np.abs(fshift_filtered) + 1)
            st.image(magnitude_filtered, caption="필터링된 스펙트럼", use_container_width=True, clamp=True)
        with col3:
            st.image(image_back, caption="복원된 이미지", use_container_width=True, clamp=True)

    else:
        st.warning("먼저 이미지를 업로드하세요.")


def page_image_pyramid():
    """페이지 8: 이미지 피라미드"""
    st.header("이미지 피라미드")

    if 'color_image' not in st.session_state:
        color_img = load_color_image()
        if color_img is not None:
            st.session_state['color_image'] = color_img

    if 'color_image' in st.session_state and st.session_state['color_image'] is not None:
        image_rgb = st.session_state['color_image']

        pyramid_type = st.selectbox("피라미드 유형", ["가우시안 피라미드", "라플라시안 피라미드"])
        levels = st.slider("레벨 수", 2, 6, 5)

        if pyramid_type == "가우시안 피라미드":
            st.subheader("가우시안 피라미드")

            # 가우시안 피라미드 생성
            pyramid = [image_rgb]
            temp_img = image_rgb.copy()
            for i in range(levels - 1):
                temp_img = cv2.pyrDown(temp_img)
                pyramid.append(temp_img)

            # 결과 표시
            cols = st.columns(levels)
            for i, col in enumerate(cols):
                with col:
                    col.image(pyramid[i], caption=f"Level {i + 1}", use_container_width=True)

        else:  # 라플라시안 피라미드
            st.subheader("라플라시안 피라미드")

            # 가우시안 피라미드 생성
            g_pyramid = [image_rgb]
            temp_img = image_rgb.copy()
            for i in range(levels - 1):
                temp_img = cv2.pyrDown(temp_img)
                g_pyramid.append(temp_img)

            # 라플라시안 피라미드 생성
            l_pyramid = []
            for i in range(len(g_pyramid) - 1):
                next_level = cv2.pyrUp(g_pyramid[i + 1])
                if next_level.shape[0] > g_pyramid[i].shape[0]:
                    next_level = next_level[:-1, :, :]
                if next_level.shape[1] > g_pyramid[i].shape[1]:
                    next_level = next_level[:, :-1, :]
                if next_level.shape != g_pyramid[i].shape:
                    next_level = cv2.resize(next_level, (g_pyramid[i].shape[1], g_pyramid[i].shape[0]))
                lap = cv2.subtract(g_pyramid[i], next_level)
                l_pyramid.append(lap)
            l_pyramid.append(g_pyramid[-1])

            # 결과 표시 - 가로로 연결
            if len(l_pyramid) > 0:
                min_height = min([img.shape[0] for img in l_pyramid])
                concatenated = cv2.resize(l_pyramid[0],
                                        (int(l_pyramid[0].shape[1] * min_height / l_pyramid[0].shape[0]), min_height))

                for img in l_pyramid[1:]:
                    resized_img = cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height))
                    concatenated = cv2.hconcat([concatenated, resized_img])

                st.image(concatenated, caption="라플라시안 피라미드 (Level 1 ~ Level " + str(levels) + ")",
                        use_container_width=True)

    else:
        st.warning("이미지를 로드할 수 없습니다.")


def page_edge_detection():
    """페이지 9: 경계 검출"""
    st.header("이미지 경계 검출")

    if 'current_image' not in st.session_state or st.session_state['current_image'] is None:
        image = load_default_image()
        if image is not None:
            st.session_state['current_image'] = image

    if 'current_image' in st.session_state and st.session_state['current_image'] is not None:
        image = st.session_state['current_image']

        edge_method = st.selectbox("경계 검출 방법", ["캐니 에지 검출기", "소벨 연산자", "프리윗 연산자"])

        if edge_method == "캐니 에지 검출기":
            st.subheader("캐니 에지 검출기 (Canny Edge Detection)")

            # 가우시안 블러
            blur_kernel = st.slider("가우시안 블러 커널 크기", 1, 11, 5, 2)
            blur_sigma = st.slider("가우시안 σ", 0.5, 5.0, 1.4, 0.1)
            blurred_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), blur_sigma)

            # 캐니 에지 검출
            threshold1 = st.slider("임계값 1", 0, 255, 50)
            threshold2 = st.slider("임계값 2", 0, 255, 150)
            canny_edges = cv2.Canny(blurred_image, threshold1=threshold1, threshold2=threshold2)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(image, caption="원본", use_container_width=True, clamp=True)
            with col2:
                st.image(blurred_image, caption="가우시안 블러", use_container_width=True, clamp=True)
            with col3:
                st.image(canny_edges, caption="캐니 에지", use_container_width=True, clamp=True)

            # 다양한 임계값 비교
            if st.checkbox("다양한 임계값 비교"):
                st.subheader("다양한 임계값 조합")
                thresholds = [(10, 50), (50, 100), (100, 150), (150, 200)]
                cols = st.columns(4)

                for i, (t1, t2) in enumerate(thresholds):
                    canny_result = cv2.Canny(blurred_image, threshold1=t1, threshold2=t2)
                    with cols[i]:
                        st.image(canny_result, caption=f"({t1}, {t2})", use_container_width=True, clamp=True)

        elif edge_method == "소벨 연산자":
            st.subheader("소벨 연산자 (Sobel Operator)")

            kernel_size = st.slider("커널 크기", 1, 7, 3, 2)

            # 소벨 커널 적용
            gx_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
            gy_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

            # 절댓값 변환 및 정규화
            gx_sobel = cv2.convertScaleAbs(gx_sobel)
            gy_sobel = cv2.convertScaleAbs(gy_sobel)

            # 결과 이미지
            sobel_result = cv2.addWeighted(gx_sobel, 0.5, gy_sobel, 0.5, 0)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.image(image, caption="원본", use_container_width=True, clamp=True)
            with col2:
                st.image(gx_sobel, caption="X 방향", use_container_width=True, clamp=True)
            with col3:
                st.image(gy_sobel, caption="Y 방향", use_container_width=True, clamp=True)
            with col4:
                st.image(sobel_result, caption="소벨 결과", use_container_width=True, clamp=True)

        else:  # 프리윗 연산자
            st.subheader("프리윗 연산자 (Prewitt Operator)")

            # 프리윗 커널 정의
            kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

            # 프리윗 커널 적용
            gx = cv2.filter2D(image, -1, kx)
            gy = cv2.filter2D(image, -1, ky)

            # 결과 이미지
            prewitt_result = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.image(image, caption="원본", use_container_width=True, clamp=True)
            with col2:
                st.image(gx, caption="X 방향", use_container_width=True, clamp=True)
            with col3:
                st.image(gy, caption="Y 방향", use_container_width=True, clamp=True)
            with col4:
                st.image(prewitt_result, caption="프리윗 결과", use_container_width=True, clamp=True)

    else:
        st.warning("먼저 이미지를 업로드하세요.")
