"""
Script to merge external datasets into the main waste dataset.
Helps in expanding the training data for better accuracy.
"""

import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm

def merge_datasets(source_dir: str, target_dir: str = 'datasets/waste'):
    """
    Merge images from source_dir into target_dir.
    Assumes source_dir has subdirectories named after classes (e.g., 'plastic', 'glass').
    
    Args:
        source_dir: Path to the external dataset
        target_dir: Path to the main dataset (default: 'datasets/waste')
    """
    
    print("=" * 70)
    print("  🔄 Dataset Merger Tool")
    print("=" * 70)
    
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    if not os.path.exists(target_dir):
        print(f"⚠️  Target directory not found. Creating: {target_dir}")
        os.makedirs(target_dir)
        
    # valid extensions
    valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    
    # Get classes from source
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Found classes in source: {classes}")
    
    total_moved = 0
    total_files = 0
    
    for class_name in classes:
        src_class_path = os.path.join(source_dir, class_name)
        tgt_class_path = os.path.join(target_dir, class_name)
        
        # Standardize class name (lowercase)
        tgt_class_path = os.path.join(target_dir, class_name.lower())
        
        os.makedirs(tgt_class_path, exist_ok=True)
        
        # Collect files
        files = []
        for ext in valid_exts:
            files.extend(glob.glob(os.path.join(src_class_path, ext)))
            files.extend(glob.glob(os.path.join(src_class_path, ext.upper())))
            
        print(f"\nProcessing '{class_name}': Found {len(files)} images")
        
        for file_path in tqdm(files, desc=f"  Merging {class_name}", leave=False):
            filename = os.path.basename(file_path)
            
            # Avoid overwriting by adding prefix if needed
            target_file = os.path.join(tgt_class_path, filename)
            if os.path.exists(target_file):
                name, ext = os.path.splitext(filename)
                timestamp = int(os.path.getmtime(file_path))
                new_name = f"{name}_{timestamp}{ext}"
                target_file = os.path.join(tgt_class_path, new_name)
            
            try:
                shutil.copy2(file_path, target_file)
                total_moved += 1
            except Exception as e:
                print(f"  ❌ Error copying {filename}: {e}")
                
    print("\n" + "=" * 70)
    print(f"  ✅ Merge Complete!")
    print(f"  Total images merged: {total_moved}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Merge datasets')
    parser.add_argument('source', help='Path to source dataset directory')
    parser.add_argument('--target', default='datasets/waste', help='Path to target dataset directory')
    
    args = parser.parse_args()
    
    merge_datasets(args.source, args.target)
