"""
Download and prepare the TrashNet dataset for waste classification training.
TrashNet: https://github.com/garythung/trashnet
"""

import os
import requests
import zipfile
from tqdm import tqdm
import shutil
from pathlib import Path

def download_file(url: str, destination: str) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_trashnet_dataset(base_dir: str = 'datasets') -> None:
    """
    Download and extract TrashNet dataset.
    
    Dataset contains 2527 images across 6 categories:
    - cardboard (403 images)
    - glass (501 images)
    - metal (410 images)
    - paper (594 images)
    - plastic (482 images)
    - trash (137 images)
    """
    print("🗑️  Downloading TrashNet Dataset...")
    
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    dataset_path = os.path.join(base_dir, 'waste')
    
    # TrashNet dataset URL (using direct download link)
    # Note: This is a mirror of the original dataset
    dataset_url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    zip_path = os.path.join(base_dir, 'trashnet.zip')
    
    # Download dataset
    if not os.path.exists(zip_path):
        print(f"📥 Downloading from {dataset_url}...")
        try:
            download_file(dataset_url, zip_path)
            print("✅ Download complete!")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("\n📝 Manual download instructions:")
            print("1. Visit: https://github.com/garythung/trashnet")
            print("2. Download the dataset-resized.zip file")
            print(f"3. Place it in: {os.path.abspath(base_dir)}")
            return
    else:
        print("✅ Dataset already downloaded!")
    
    # Extract dataset
    if not os.path.exists(dataset_path):
        print(f"📦 Extracting dataset to {dataset_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            
            # Reorganize if needed
            extracted_dir = os.path.join(base_dir, 'dataset-resized')
            if os.path.exists(extracted_dir):
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)
                shutil.move(extracted_dir, dataset_path)
            
            print("✅ Extraction complete!")
        except Exception as e:
            print(f"❌ Error extracting dataset: {e}")
            return
    else:
        print("✅ Dataset already extracted!")
    
    # Verify dataset structure
    print("\n📊 Dataset Statistics:")
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    total_images = 0
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            num_images = len([f for f in os.listdir(category_path) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
            total_images += num_images
            print(f"  {category.capitalize():12} : {num_images:4} images")
        else:
            print(f"  ⚠️  {category.capitalize():12} : Missing!")
    
    print(f"\n  Total        : {total_images:4} images")
    
    if total_images > 0:
        print("\n✅ Dataset ready for training!")
        print(f"📁 Dataset location: {os.path.abspath(dataset_path)}")
    else:
        print("\n❌ Dataset verification failed!")

def download_alternative_dataset(base_dir: str = 'datasets') -> None:
    """
    Alternative: Download waste classification dataset from Kaggle.
    Requires Kaggle API credentials.
    """
    print("\n🔄 Alternative: Using Kaggle dataset...")
    print("📝 To use Kaggle datasets:")
    print("1. Install kaggle: pip install kaggle")
    print("2. Setup API credentials: https://www.kaggle.com/docs/api")
    print("3. Run: kaggle datasets download -d techsash/waste-classification-data")
    print("\nOr use the TrashNet dataset from GitHub (recommended)")

if __name__ == '__main__':
    print("=" * 60)
    print("  Waste Classification Dataset Downloader")
    print("=" * 60)
    
    # Download TrashNet dataset
    download_trashnet_dataset()
    
    print("\n" + "=" * 60)
    print("  Next Steps:")
    print("=" * 60)
    print("1. Run: python train_model.py")
    print("2. Wait for training to complete (~20-30 minutes)")
    print("3. Run: python run.py")
    print("4. Test the improved model!")
