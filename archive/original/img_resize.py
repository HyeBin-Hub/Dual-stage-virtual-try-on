import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm

data_root = "./Dresscode-512/test/skeletons"  
save_root = "./Dresscode-256/test/skeletons"     

# img = Image.open(data_root).convert('RGB')

# resize_img = img.resize((192, 256))

# resize_img.save(save_root)

if not os.path.exists(save_root):
    os.mkdir(save_root)

for cloth_file_name in tqdm(os.listdir(data_root), desc="Processing Images"):                                                           
    
    # output_cloth_file_name = cloth_file_name.split("_5.jpg")[0] + "_0.jpg"
    
    cloth_img_path = os.path.join(data_root, cloth_file_name)
    
    img = Image.open(cloth_img_path).convert('RGB')
    
    resize_img = img.resize((192, 256))
    
    save_path = os.path.join(save_root, cloth_file_name)
    
    resize_img.save(save_path)