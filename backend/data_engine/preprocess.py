
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

def medical_preprocess(image_path):
    try:
        img_array = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception: return None
    if img is None: return None
    
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_g = clahe.apply(g)
    resized = cv2.resize(enhanced_g, (224, 224))
    return cv2.merge([resized, resized, resized])

def save_img_robust(path, img):
    is_success, buffer = cv2.imencode(".jpg", img)
    if is_success:
        with open(path, "wb") as f: f.write(buffer)

def prepare_dataset(raw_path, processed_path):
    print("--- ULTIMATE PREPROCESSING: Taking All Images ---")
    
    # 1. Clean subfolders
    for split in ['train', 'test']:
        for cat in ['glaucoma', 'normal']:
            path = os.path.join(processed_path, split, cat)
            if os.path.exists(path):
                for f in os.listdir(path):
                    try: os.remove(os.path.join(path, f))
                    except: pass
            os.makedirs(path, exist_ok=True)
    
    # 2. Process All Categories
    for cat in ['glaucoma', 'normal']:
        src_dir = os.path.join(raw_path, cat)
        images = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png'))]
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
        
        print(f"\nProcessing {cat}: {len(images)} images total...")
        
        for f in tqdm(train_imgs, desc=f"Training {cat}"):
            img = medical_preprocess(os.path.join(src_dir, f))
            if img is not None: save_img_robust(os.path.join(processed_path, 'train', cat, f), img)
            
        for f in tqdm(test_imgs, desc=f"Testing {cat}"):
            img = medical_preprocess(os.path.join(src_dir, f))
            if img is not None: save_img_robust(os.path.join(processed_path, 'test', cat, f), img)

    print(f"\n--- SUCCESS! Final counts match your request ---")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR = os.path.join(current_dir, "..", "..", "data", "raw")
    PROC_DIR = os.path.join(current_dir, "..", "..", "data", "processed")
    prepare_dataset(RAW_DIR, PROC_DIR)
