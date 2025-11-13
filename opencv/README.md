# OpenCV 이미지 처리 Streamlit 앱

OpenCV를 사용한 인터랙티브 이미지 처리 웹 애플리케이션

## 📁 파일 구조

```
opencv/
├── opencv_streamlit_app.py    # 메인 앱 파일 (라우팅)
├── utils.py                    # 유틸리티 함수 모음
├── pages_basic.py              # 1장: OpenCV 기초 기능
├── pages_advanced.py           # 2장: 이미지 처리 기초 기능
├── like_lenna.png             # 샘플 이미지
├── requirements.txt           # 필요한 패키지 목록
└── README.md                  # 이 파일
```

## 📦 설치 방법

### 1. 필요한 패키지 설치

```bash
cd opencv
pip install -r requirements.txt
```

### 2. 앱 실행

```bash
streamlit run opencv_streamlit_app.py
```

## 📚 기능 목록

### 1장: OpenCV 기초
1. **이미지 업로드 및 보기**
   - 이미지 업로드 기능
   - 이미지 정보 표시 (shape, dtype)

2. **이미지 변환**
   - 크기 변환 (픽셀 지정 / 배율 지정)
   - 대칭 변환 (상하, 좌우, 상하좌우)
   - 회전 변환 (각도 및 배경 색상 조절)
   - 이미지 자르기

3. **도형 그리기**
   - 선, 원, 사각형, 타원
   - 다각형 (외곽선 / 채우기)
   - 격자 그리기

### 2장: 이미지 처리 기초
4. **색 공간 변환**
   - RGB to Grayscale (공식 설명 포함)
   - RGB to HSV (채널 분리)

5. **정규화 및 표준화**
   - 정규화 (Normalization): 0~1 범위로 변환
   - 표준화 (Standardization): 평균 0, 표준편차 1

6. **노이즈 및 필터링**
   - Salt & Pepper 노이즈 + 중앙값 필터링
   - Gaussian 노이즈 + 가우시안 필터링

7. **푸리에 변환**
   - 2D FFT 및 주파수 스펙트럼
   - High-pass / Low-pass 필터링

8. **이미지 피라미드**
   - 가우시안 피라미드
   - 라플라시안 피라미드

9. **경계 검출**
   - 캐니 에지 검출기 (Canny)
   - 소벨 연산자 (Sobel)
   - 프리윗 연산자 (Prewitt)

## 📦 필요한 라이브러리 (requirements.txt)

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| streamlit | >= 1.28.0 | 웹 앱 프레임워크 |
| opencv-python | >= 4.8.0 | 이미지 처리 |
| numpy | >= 1.24.0 | 수치 연산 및 배열 처리 |
| Pillow | >= 10.0.0 | 이미지 파일 입출력 |
| matplotlib | >= 3.7.0 | 이미지 시각화 (선택사항) |

설치 명령어:
```bash
pip install -r requirements.txt
```

## 🔧 코드 구조

### utils.py
모든 페이지에서 공통으로 사용하는 헬퍼 함수들:
- `load_default_image()`: Grayscale 이미지 로드
- `load_color_image()`: RGB 이미지 로드
- `generate_salt_noise()`: 소금 노이즈 생성
- `generate_pepper_noise()`: 후추 노이즈 생성
- 이미지 변환 함수들

### pages_basic.py
1장 OpenCV 기초 기능의 페이지 함수들:
- `page_image_upload()`: 페이지 1
- `page_image_transform()`: 페이지 2
- `page_draw_shapes()`: 페이지 3

### pages_advanced.py
2장 이미지 처리 기초 기능의 페이지 함수들:
- `page_color_space()`: 페이지 4
- `page_normalization()`: 페이지 5
- `page_noise_filtering()`: 페이지 6
- `page_fourier_transform()`: 페이지 7
- `page_image_pyramid()`: 페이지 8
- `page_edge_detection()`: 페이지 9

### opencv_streamlit_app.py
메인 앱 파일로 페이지 라우팅을 담당:
- Streamlit 페이지 설정
- 사이드바 메뉴
- 페이지 선택에 따른 함수 호출

## 💡 사용 팁

1. 첫 실행 시 "이미지 업로드 및 보기" 페이지에서 이미지를 로드하세요
2. 각 기능은 독립적으로 작동하며, session_state를 통해 이미지를 공유합니다
3. 슬라이더를 조절하여 실시간으로 결과를 확인할 수 있습니다

## 📝 참고

이 앱은 이미지 처리 교육용으로 제작되었습니다.
- 1장: OpenCV 기본 사용법
- 2장: 이미지 처리 이론 및 실습
