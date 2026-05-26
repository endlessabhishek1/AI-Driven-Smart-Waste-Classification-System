import os
import cv2
import hashlib
from pathlib import Path
from tqdm import tqdm
import argparse

def get_image_hash(filepath):
    """Generate MD5 hash for a file to detect exact duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def clean_dataset(dataset_dir: str, remove_corrupted: bool = True, remove_duplicates: bool = True):
    """
    Clean dataset by removing corrupted images and exact duplicates.
    This ensures training stability and prevents data leakage.
    """
    print("=" * 70)
    print("  🧹 Waste AI Dataset Cleaner")
    print("=" * 70)
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Directory not found: {dataset_dir}")
        return

    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    corrupted_count = 0
    duplicate_count = 0
    total_images = 0
    
    # Track hashes per class to avoid cross-class duplicate removal (which might be mislabeled data,
    # but exact duplicate removal within same class is safe)
    # We can also track global hashes to flag cross-class duplicates (data leakage/conflict).
    global_hashes = {} 

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        total_images += len(images)
        
        print(f"\nProcessing '{class_name}' ({len(images)} images)...")
        
        for img_name in tqdm(images, desc="Cleaning", leave=False):
            img_path = os.path.join(class_dir, img_name)
            
            # 1. Check for corruption
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise Exception("cv2.imread returned None")
            except Exception as e:
                if remove_corrupted:
                    os.remove(img_path)
                    corrupted_count += 1
                else:
                    print(f"⚠️ Corrupted image found: {img_path}")
                continue
                
            # 2. Check for duplicates
            if remove_duplicates:
                img_hash = get_image_hash(img_path)
                
                if img_hash in global_hashes:
                    # Duplicate found
                    conflict_class = global_hashes[img_hash]
                    if conflict_class == class_name:
                        # Duplicate within same class -> safe to remove
                        os.remove(img_path)
                        duplicate_count += 1
                    else:
                        # Duplicate across different classes -> Data leakage/conflict!
                        print(f"\n🚨 CONFLICT: {img_name} in '{class_name}' is identical to an image in '{conflict_class}'!")
                        print(f"Removing duplicate from '{class_name}' to prevent confusion.")
                        os.remove(img_path)
                        duplicate_count += 1
                else:
                    global_hashes[img_hash] = class_name

    print("\n" + "=" * 70)
    print("  ✅ Cleaning Complete!")
    print("=" * 70)
    print(f"  Total Initial Images : {total_images}")
    print(f"  Corrupted Removed    : {corrupted_count}")
    print(f"  Duplicates Removed   : {duplicate_count}")
    print(f"  Final Valid Images   : {total_images - corrupted_count - duplicate_count}")
    print("=" * 70)
    
    if (corrupted_count + duplicate_count) > 0:
        print("Dataset quality improved! This will help achieve your 80-95% accuracy goal.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean dataset by removing corrupted files and duplicates.')
    parser.add_argument('--dataset', type=str, default='datasets/waste', help='Path to dataset directory')
    parser.add_argument('--keep-corrupted', action='store_true', help='Do not remove corrupted files (just warn)')
    parser.add_argument('--keep-duplicates', action='store_true', help='Do not remove exact duplicates')
    
    args = parser.parse_args()
    
    clean_dataset(
        dataset_dir=args.dataset,
        remove_corrupted=not args.keep_corrupted,
        remove_duplicates=not args.keep_duplicates
    )
