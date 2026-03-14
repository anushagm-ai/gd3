import os

def check_dataset(data_path):
    print("--- Retinal Data Diagnostic (V2) ---")
    abs_path = os.path.abspath(data_path)
    print(f"Checking path: {abs_path}")
    
    if not os.path.exists(abs_path):
        print(f"Error: Folder '{abs_path}' not found.")
        print("Please place your downloaded dataset in 'data/raw'.")
        return

    files = []
    for root, dirs, filenames in os.walk(abs_path):
        for f in filenames:
            if f.endswith(('.jpg', '.png', '.jpeg')):
                files.append(f)
    
    print(f"Total Images Found: {len(files)}")
    if len(files) > 0:
        print("\nSuccess! Dataset detected. Next Step: Categorization and Medical Preprocessing.")
    else:
        print("\nFolder is empty. Please unzip your Kaggle dataset here.")

if __name__ == "__main__":
    # Correct relative path from this script location to the data/raw folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, "..", "..", "data", "raw")
    check_dataset(raw_data_dir)
