import cv2
import os
import numpy as np


# 이미지 데이터셋 폴더 경로
input_dir_val_normal = "C:\\Users\\myong\\Desktop\\x-ray\\image input\\normal"  # 전처리전 정상 데이터 폴더
input_dir_val_pneumonia  = "C:\\Users\\myong\\Desktop\\x-ray\\image input\\pneumonia"  # 전처리전 정상 데이터 폴더

output_dir_val_normal = "C:\\Users\\myong\\Desktop\\x-ray\\image output\\normal"  # 전처리된 폐렴용 데이터 저장할 폴더
output_dir_val_pneumonia = "C:\\Users\\myong\\Desktop\\x-ray\\image output\\pneumonia"  # 전처리된 폐렴용 데이터 저장할 폴더

# 이미지 크기 설정
img_width = 256
img_height = 256

# 폴더가 없을 경우 생성
if not os.path.exists(output_dir_val_normal):
    os.makedirs(output_dir_val_normal)

if not os.path.exists(output_dir_val_pneumonia):
    os.makedirs(output_dir_val_pneumonia)


# 이미지 정제 함수
def preprocess_image(img_path):

    img = cv2.imread(img_path) # 이미지 로드
    
    resized_img = cv2.resize(img, (img_width, img_height)) # 이미지 리사이즈
    
    denoised_img = cv2.GaussianBlur(resized_img, (5, 5), 0) # 노이즈 제거

    return denoised_img

# 데이터셋 내 모든 이미지 처리
for img_name in os.listdir(input_dir_val_normal): #정상용
    img_path = os.path.join(input_dir_val_normal, img_name)
    
    # 이미지가 .jpg 또는 .png 또는 .jpeg 파일인 경우만 처리
    if img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('.jpeg'):
        processed_img = preprocess_image(img_path)
        
        # 처리된 이미지를 새로운 경로에 저장
        output_path = os.path.join(output_dir_val_normal, img_name)
        cv2.imwrite(output_path, processed_img)

for img_name in os.listdir(input_dir_val_pneumonia): # 폐렴용
    img_path = os.path.join(input_dir_val_pneumonia, img_name)
    
    # 이미지가 .jpg 또는 .png 또는 .jpeg 파일인 경우만 처리
    if img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('.jpeg'):
        processed_img = preprocess_image(img_path)
        
        # 처리된 이미지를 새로운 경로에 저장
        output_path = os.path.join(output_dir_val_pneumonia, img_name)
        cv2.imwrite(output_path, processed_img)