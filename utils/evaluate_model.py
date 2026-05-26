"""
Model Testing and Error Analysis Script for Waste Classification

This script evaluates a trained Waste AI model on a dataset, computes precision, 
recall, F1-score, and accuracy, generates a confusion matrix, and performs 
error analysis by saving misclassified images into categorized folders.
"""

import os
import cv2
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
# Add parent directory to path to allow importing app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai_model import WasteClassifier

def evaluate_and_analyze_errors(dataset_dir: str, model_path: str, output_dir: str = 'error_analysis'):
    """
    Evaluates the model and isolates wrong predictions for analysis.
    
    Args:
        dataset_dir: Path to the test dataset (should contain class subdirectories).
        model_path: Path to the trained .pth model file.
        output_dir: Directory to save misclassified images and metrics.
    """
    print("=" * 70)
    print("  🧪 Waste Classification - Model Testing & Error Analysis")
    print("=" * 70)
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset not found: {dataset_dir}")
        return
        
    # Initialize the model
    print(f"Loading model from: {model_path}...")
    classifier = WasteClassifier(model_path=model_path)
    class_names = classifier.class_names
    
    # Prepare output directories
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    
    # Variables to track performance
    y_true = []
    y_pred = []
    misclassified_count = 0
    total_images = 0
    
    # Evaluate each class directory
    dataset_classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print("\n🔍 Running inference on dataset...")
    for true_class in dataset_classes:
        if true_class not in class_names:
            print(f"⚠️ Skipping unknown class directory: {true_class}")
            continue
            
        class_dir = os.path.join(dataset_dir, true_class)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Evaluating {true_class}", leave=False):
            img_path = os.path.join(class_dir, img_name)
            
            # Read and predict
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            total_images += 1
            prediction = classifier.predict(img)
            pred_class = prediction['class']
            
            # Track for metrics
            y_true.append(true_class)
            y_pred.append(pred_class)
            
            # Error Analysis: Save misclassified images
            if true_class != pred_class:
                misclassified_count += 1
                
                # Create specific error directory: e.g., "paper_as_cardboard"
                error_type_dir = os.path.join(output_dir, f"{true_class}_as_{pred_class}")
                os.makedirs(error_type_dir, exist_ok=True)
                
                # Save the image with confidence score in filename
                conf = prediction['confidence']
                new_filename = f"conf_{conf:.1f}_{img_name}"
                dest_path = os.path.join(error_type_dir, new_filename)
                
                # Copy image instead of moving to preserve the original dataset
                shutil.copy2(img_path, dest_path)
                
    # Calculate Metrics
    if total_images == 0:
        print("❌ No valid images found in the dataset.")
        return
        
    print("\n" + "=" * 70)
    print("  📊 Evaluation Results")
    print("=" * 70)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Total Images: {total_images}")
    print(f"Misclassified: {misclassified_count}")
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    
    # Save Report to File
    with open(report_path, 'w') as f:
        f.write("Waste Classification Evaluation Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset: {dataset_dir}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Total Images Tested: {total_images}\n")
        f.write(f"Total Misclassified: {misclassified_count}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"📄 Report saved to: {report_path}")

    # Generate Confusion Matrix
    print("\n🎨 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Waste Classification Confusion Matrix', pad=20, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    print(f"📈 Confusion Matrix saved to: {cm_path}")
    print(f"🔎 Error Analysis details saved in: {os.path.abspath(output_dir)}")
    print("\n✅ Evaluation Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Waste AI model and analyze errors.')
    parser.add_argument('--dataset', type=str, default='datasets/waste', 
                        help='Path to the dataset directory (e.g., datasets/test)')
    parser.add_argument('--model', type=str, default='models/waste_classifier.pth', 
                        help='Path to the trained model (.pth)')
    parser.add_argument('--output', type=str, default='error_analysis', 
                        help='Directory to save analysis results and misclassified images')
    
    args = parser.parse_args()
    
    evaluate_and_analyze_errors(
        dataset_dir=args.dataset,
        model_path=args.model,
        output_dir=args.output
    )
