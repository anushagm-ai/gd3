
import tensorflow as tf
import os
import numpy as np
import cv2

def test_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "glaucoma_model_v2.h5")
    test_dir = os.path.join(current_dir, "..", "..", "data", "processed", "test")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    print("--- Loading Model for Diagnostic Test ---")
    model = tf.keras.models.load_model(model_path)
    
    # Detect folders (Glaucoma, Normal)
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    print(f"Clinical Classes: {classes}")
    
    for category in classes:
        cat_path = os.path.join(test_dir, category)
        images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n--- Testing Category: {category.upper()} (Total: {len(images)} images) ---")
        for img_name in images:
            img_path = os.path.join(cat_path, img_name)
            
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None: continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0)
            
            preds = model.predict(img, verbose=0)
            print(f"Image: {img_name} | Glaucoma(Ind0): {preds[0][0]:.4f} | Normal(Ind1): {preds[0][1]:.4f}")

if __name__ == "__main__":
    test_model()
