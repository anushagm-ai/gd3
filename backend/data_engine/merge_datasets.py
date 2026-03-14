
import os
import shutil
import pandas as pd
import json
from tqdm import tqdm

def merge_datasets():
    base_path = r"c:\Users\Anusha\OneDrive\文档\gd1\data\raw"
    archive_path = os.path.join(base_path, "archive")
    
    glaucoma_dest = os.path.join(base_path, "glaucoma")
    normal_dest = os.path.join(base_path, "normal")
    
    # Reset folders to start fresh
    if os.path.exists(glaucoma_dest): shutil.rmtree(glaucoma_dest)
    if os.path.exists(normal_dest): shutil.rmtree(normal_dest)
    os.makedirs(glaucoma_dest, exist_ok=True)
    os.makedirs(normal_dest, exist_ok=True)
    
    stats = {"glaucoma": 0, "normal": 0}
    NORMAL_CAP = 2000 # Limit normal images to match glaucoma for balance

    def copy_img(src, filename, dataset_name, folder_type, label):
        if label == 1:
            target = glaucoma_dest
            stats["glaucoma"] += 1
        else:
            if stats["normal"] >= NORMAL_CAP:
                return # Stop taking normal images
            target = normal_dest
            stats["normal"] += 1
        
        new_name = f"{dataset_name}_{folder_type}_{filename}".lower().replace("\\", "_").replace("/", "_")
        shutil.copy2(src, os.path.join(target, new_name))

    print(f"--- BALANCED EXTRACTION (Capping Normal at {NORMAL_CAP}) ---")

    # 1. G1020
    g1020_csv = os.path.join(archive_path, "G1020", "G1020.csv")
    if os.path.exists(g1020_csv):
        df = pd.read_csv(g1020_csv)
        g1020_root = os.path.join(archive_path, "G1020")
        for fld in ["Images", "Images_Square", "Images_Cropped/img", "NerveRemoved_Images"]:
            img_dir = os.path.join(g1020_root, fld)
            if os.path.exists(img_dir):
                for _, row in df.iterrows():
                    copy_img(os.path.join(img_dir, row['imageID']), row['imageID'], "g1020", fld.replace("/","_"), int(row['binaryLabels']))

    # 2. ORIGA
    origa_csv = os.path.join(archive_path, "ORIGA", "origa_info.csv")
    if os.path.exists(origa_csv):
        df = pd.read_csv(origa_csv)
        origa_root = os.path.join(archive_path, "ORIGA")
        for fld in ["Images", "Images_Cropped", "Images_Square"]:
            img_dir = os.path.join(origa_root, fld)
            if os.path.exists(img_dir):
                for _, row in df.iterrows():
                    img_name = os.path.basename(row['Image'])
                    copy_img(os.path.join(img_dir, img_name), img_name, "origa", fld, int(row['Label']))

    # 3. REFUGE
    for subset in ["train", "val", "test"]:
        sub_path = os.path.join(archive_path, "REFUGE", subset)
        idx_f = os.path.join(sub_path, "index.json")
        if os.path.exists(idx_f):
            with open(idx_f, 'r') as f:
                data = json.load(f)
            for fld in ["Images", "Images_Cropped"]:
                img_dir = os.path.join(sub_path, fld)
                if os.path.exists(img_dir):
                    for k in data:
                        nm = data[k]['ImgName']
                        lbl = data[k].get('Label', 1 if nm.startswith('g') else 0)
                        copy_img(os.path.join(img_dir, nm), nm, "refuge", f"{subset}_{fld}", lbl)

    print("\n--- FINAL BALANCED SUMMARY ---")
    print(f"Glaucoma images: {stats['glaucoma']} (Maximum found)")
    print(f"Normal images:   {stats['normal']} (Capped for balance)")
    print(f"Total dataset:   {sum(stats.values())}")

if __name__ == "__main__":
    merge_datasets()
