import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def create_binary_mask_pil(image_path, output_path="binary_mask.png"):
    # 이미지 로드 (RGBA 지원)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # RGBA 이미지라면 알파 채널 사용
    if image.shape[-1] == 4:
        alpha_channel = image[:, :, 3]  # 알파 채널 가져오기
        _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
    
    else:
        # 배경 제거를 위한 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Otsu Threshold 적용 (옷이 흰색이 되도록 반전)
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 노이즈 제거 (Morphology 연산)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # 작은 구멍 메움
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)   # 작은 노이즈 제거

    # PIL로 변환 및 저장
    mask_pil = Image.fromarray(binary_mask)
    mask_pil = mask_pil.convert("L")  # Grayscale로 변환
    mask_pil.save(output_path, format="JPEG", quality=95)

    print(f"✅ Binary mask saved at: {output_path}")


# 사용 예시
# image_path = "./im_c/000000_0.jpg"  # 입력 이미지 경로
# output_path = "binary_mask-3.png"  # 결과 저장 경로

for img_name in os.listdir("./Dresscode-512/test/agnostic-v3.2"):
    img_path = os.path.join("./Dresscode-512/test/agnostic-v3.2", img_name)
    
    output_path = os.path.join("./Dresscode-256/test/agnostic-v3.2",img_name)
    create_binary_mask_pil(img_path, output_path)


