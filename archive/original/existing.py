import os

def get_image_files(directory):
    """해당 디렉토리에서 이미지 파일 목록을 가져옴"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return {file for file in os.listdir(directory) if os.path.splitext(file)[1].lower() in image_extensions}

def compare_directories(a_dir, b_dir):
    """a 디렉토리의 이미지들이 b 디렉토리에 존재하는지 확인"""
    a_images = get_image_files(a_dir)
    b_images = get_image_files(b_dir)
    
    missing_images = a_images - b_images  # b 디렉토리에 없는 이미지들
    
    if not missing_images:
        print("모든 이미지가 b 디렉토리에 존재합니다.")
    else:
        print("다음 이미지들이 b 디렉토리에 존재하지 않습니다:")
        for img in missing_images:
            print(img)

# 사용 예시
b_directory = "./result-dress/test_unpaired/test/c_result"  # a 디렉토리 경로 설정
a_directory = "./Dresscode-512/test/cloth_mask_deformation"  # b 디렉토리 경로 설정
compare_directories(a_directory, b_directory)