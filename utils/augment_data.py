
import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import argparse
import shutil

def augment_dataset(source_dir: str, target_dir: str = None, multipliers: int = 3):
    """
    Augment dataset by generating synthetic variations of existing images.
    
    Args:
        source_dir: Directory containing class subdirectories of images
        target_dir: Directory to save augmented images (if None, saves in source_dir)
        multipliers: Number of augmented versions to generate per image
    """
    
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Define augmentation pipeline
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=1),
        ], p=0.2),
        A.CLAHE(clip_limit=2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, p=0.5),
    ])

    print(f"🚀 Starting data augmentation (x{multipliers})...")
    
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        target_class_path = os.path.join(target_dir, class_name) if target_dir else class_path
        
        if target_dir:
            os.makedirs(target_class_path, exist_ok=True)
            
        print(f"Processing class: {class_name}")
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=class_name, leave=False):
            img_path = os.path.join(class_path, img_name)
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Copy original if generating to new directory
            if target_dir:
                shutil.copy2(img_path, os.path.join(target_class_path, img_name))
            
            # Generate augmentations
            for i in range(multipliers):
                augmented = transform(image=image)['image']
                augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                # Construct new filename
                name, ext = os.path.splitext(img_name)
                new_name = f"{name}_aug_{i}{ext}"
                save_path = os.path.join(target_class_path, new_name)
                
                cv2.imwrite(save_path, augmented)

    print("\n✅ Data augmentation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Waste AI Data Augmentation Tool')
    parser.add_argument('--source', type=str, default='datasets/waste', help='Source dataset directory')
    parser.add_argument('--target', type=str, help='Target directory (optional)')
    parser.add_argument('--count', type=int, default=3, help='Number of augmentations per image')
    
    args = parser.parse_args()
    
    augment_dataset(args.source, args.target, args.count)
