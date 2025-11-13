"""
이미지 처리를 위한 유틸리티 함수 모음
"""
import cv2
import numpy as np
from PIL import Image


def load_default_image():
    """기본 이미지(like_lenna.png) 로드 - Grayscale"""
    try:
        img = cv2.imread('like_lenna.png', cv2.IMREAD_GRAYSCALE)
        return img
    except:
        return None


def load_color_image():
    """컬러 이미지 로드 (RGB)"""
    try:
        img = cv2.imread('like_lenna.png', cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except:
        return None


def numpy_to_pil(img):
    """numpy array를 PIL Image로 변환"""
    if img.ndim == 2:  # Grayscale
        return Image.fromarray(img)
    else:  # BGR to RGB
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_numpy(pil_img, grayscale=True):
    """PIL Image를 numpy array로 변환"""
    img = np.array(pil_img)
    if grayscale and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def generate_salt_noise(image, ratio=0.05):
    """소금 노이즈 생성"""
    num_salt = np.ceil(ratio * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    salted_image = image.copy()
    salted_image[coords[0], coords[1]] = 255
    return salted_image


def generate_pepper_noise(image, ratio=0.05):
    """후추 노이즈 생성"""
    num_pepper = np.ceil(ratio * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    peppered_image = image.copy()
    peppered_image[coords[0], coords[1]] = 0
    return peppered_image
