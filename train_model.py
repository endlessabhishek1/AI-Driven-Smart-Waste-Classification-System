"""
Train waste classification model using EfficientNet-B0 transfer learning.
Uses TrashNet dataset with enhanced data augmentation and deep fine-tuning for improved accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import time

def train_model(dataset_path: str = 'datasets/waste', 
                epochs: int = 20,  # EfficientNet converges faster
                batch_size: int = 16, # EfficientNet uses more memory
                learning_rate: float = 0.0003,
                model_save_path: str = 'models/waste_classifier.pth'):
    """
    Train waste classification model using transfer learning with EfficientNet-B0.
    
    Args:
        dataset_path: Path to dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        model_save_path: Path to save trained model
    """
    
    print("=" * 70)
    print("  🗑️  Waste Classification AI Enhanced Training (EfficientNet-B0)")
    print("=" * 70)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found at {dataset_path}")
        print("📥 Please ensure your dataset is in place.")
        return
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Enhanced Data Augmentation
    # EfficientNet takes 224x224 input usually (B0)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)) 
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("\n📂 Loading and processing dataset...")
    try:
        full_dataset = datasets.ImageFolder(dataset_path)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Split into train and validation
    train_size = int(0.85 * len(full_dataset)) # 85% training
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Wrapper for transforms
    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        
        def __len__(self):
            return len(self.subset)

    train_dataset = TransformSubset(train_dataset, train_transform)
    val_dataset = TransformSubset(val_dataset, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    # Print dataset info
    class_names = full_dataset.classes
    print(f"\n📊 Dataset Information:")
    print(f"  Total images    : {len(full_dataset)}")
    print(f"  Training images : {train_size}")
    print(f"  Validation images: {val_size}")
    print(f"  Classes         : {class_names}")
    
    # Initialize EfficientNet-B0
    print("\n🤖 Initializing EfficientNet-B0 model...")
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except:
        # Fallback for older torch versions
        model = models.efficientnet_b0(pretrained=True)
    
    # Unfreeze all layers for fine-tuning
    # EfficientNet benefits from fine-tuning deeper layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Replace classifier head
    # EfficientNet-B0 classifier is 'classifier' containing Sequential(Dropout, Linear)
    # in_features is 1280 for B0
    num_features = model.classifier[1].in_features 
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True), 
        nn.Linear(num_features, 512),
        nn.SiLU(), # Swish activation often used with EfficientNet
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(512, len(class_names))
    )
    
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW with slightly lower LR for fine-tuning entire net
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    print("\n🚀 Starting training...")
    print("=" * 70)
    
    try:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 70)
            
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_loader, desc='  Train', leave=False)
            for images, labels in train_bar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * train_correct / train_total:.1f}%'})
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc='  Val  ', leave=False)
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * val_correct / val_total:.1f}%'})
            
            # Metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100 * train_correct / train_total
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100 * val_correct / val_total
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            
            print(f"  Train: Loss={epoch_train_loss:.4f} | Acc={epoch_train_acc:.2f}%")
            print(f"  Val  : Loss={epoch_val_loss:.4f} | Acc={epoch_val_acc:.2f}%")
            
            # Scheduler step
            scheduler.step(epoch_val_acc)
            
            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                print(f"  ⭐ New Best Model Saved! (Acc: {best_val_acc:.2f}%)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user. Saving current progress...")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  ✅ Training Complete in {total_time/60:.1f} minutes!")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved to: {model_save_path}")
    
    plot_training_history(history)
    
    return model, history

def plot_training_history(history: dict):
    """Plot training and validation metrics."""
    try:
        plt.style.use('ggplot')
    except:
        pass
        
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Loss Change', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    plt.title('Accuracy Improvement', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'models/training_history_{timestamp}.png'
    os.makedirs('models', exist_ok=True)
    plt.savefig(plot_path)
    print(f"\n📈 Training plot saved to: {plot_path}")

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Waste Classification Model Training Script (EfficientNet-B0)")
    print("=" * 70)
    
    if not os.path.exists('datasets/waste'):
        print("\n❌ Dataset not found at 'datasets/waste'")
        print("Please check your directories.")
    else:
        train_model(
            dataset_path='datasets/waste',
            epochs=5, 
            batch_size=16,
            learning_rate=0.0003,
            model_save_path='models/waste_classifier.pth'
        )
