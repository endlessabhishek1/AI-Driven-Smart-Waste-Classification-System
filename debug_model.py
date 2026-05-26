import cv2
import os
import torch
from app.ai_model import WasteClassifier

def test_model():
    print("🔄 Loading model...")
    classifier = WasteClassifier(model_path='models/waste_classifier.pth')
    
    # Define paths to test images (one from each category if possible)
    dataset_path = 'datasets/waste'
    categories = ['paper', 'cardboard', 'plastic', 'metal', 'glass', 'trash']
    
    print("\n🧐 Testing model on dataset images...")
    
    for category in categories:
        folder_path = os.path.join(dataset_path, category)
        if not os.path.exists(folder_path):
            print(f"⚠️ Category '{category}' not found in dataset.")
            continue
            
        # Get first image
        files = os.listdir(folder_path)
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"⚠️ No images found in '{category}'.")
            continue
            
        image_path = os.path.join(folder_path, images[0])
        print(f"\nTesting {category} image: {images[0]}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            continue
            
        # Predict
        prediction = classifier.predict(image)
        predicted_class = prediction['class']
        confidence = prediction['confidence']
        
        # Result
        match_icon = "✅" if predicted_class == category else "❌"
        print(f"{match_icon} Expected: {category} | Predicted: {predicted_class} ({confidence}%)")

if __name__ == "__main__":
    test_model()
