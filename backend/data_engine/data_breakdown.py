
import os

def get_counts(base_path):
    summary = {
        "train": {"glaucoma": 0, "normal": 0},
        "test": {"glaucoma": 0, "normal": 0}
    }
    
    for split in ["train", "test"]:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            continue
            
        for category in ["glaucoma", "normal"]:
            cat_path = os.path.join(split_path, category)
            if os.path.exists(cat_path):
                # Count files with image extensions
                files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                summary[split][category] = len(files)
                
    return summary

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, "..", "..", "data", "processed")
    
    print("\n--- Scientific Dataset Breakdown (Clinical Dashboard) ---")
    if os.path.exists(processed_dir):
        counts = get_counts(processed_dir)
        
        tr_g = counts["train"]["glaucoma"]
        tr_n = counts["train"]["normal"]
        te_g = counts["test"]["glaucoma"]
        te_n = counts["test"]["normal"]
        
        print(f"\nTraining Set (Main Learning):")
        print(f"  - Glaucoma: {tr_g}")
        print(f"  - Normal:   {tr_n}")
        print(f"  - Subtotal: {tr_g + tr_n}")
        
        print(f"\nTesting Set (Performance Verification):")
        print(f"  - Glaucoma: {te_g}")
        print(f"  - Normal:   {te_n}")
        print(f"  - Subtotal: {te_g + te_n}")
        
        print(f"\n--- MISSION TOTAL ---")
        print(f"  Total Glaucoma: {tr_g + te_g}")
        print(f"  Total Normal:   {tr_n + te_n}")
        print(f"  GRAND TOTAL:    {tr_g + tr_n + te_g + te_n}")
    else:
        print(f"Error: Processed data folder not found at {processed_dir}")
