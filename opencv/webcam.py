import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 하르 캐스케이드 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 비디오 변환 클래스
class FaceDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img

st.title("실시간 웹캠 얼굴 인식")
st.write("웹캠에서 얼굴을 실시간으로 감지합니다. 'Start'를 눌러 시작하세요.")

# 웹캠 스트리밍
webrtc_streamer(key="face-detection", video_transformer_factory=FaceDetectionTransformer)