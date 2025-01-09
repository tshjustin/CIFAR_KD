import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
import torchvision.models as models

def create_teacher_model(num_classes=10):
    """Create and modify ResNet50 for CIFAR-10"""
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Modify the first conv layer to handle CIFAR-10's 32x32 images
    model.maxpool = nn.Identity() # Remove max pooling to preserve spatial dimensions for small images
    num_ftrs = model.fc.in_features # -> 10 classes 
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_teacher(model, train_loader, test_loader, config, device):
    """Finetune teacher model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    save_dir = Path(config['training']['save_dir']) / 'teacher'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_acc = 0.0
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training Teacher")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % config['training']['log_interval'] == config['training']['log_interval'] - 1:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {running_loss / config['training']['log_interval']:.3f}")
                running_loss = 0.0
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating Teacher"):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Teacher Accuracy on test set: {accuracy:.2f}%')
        
        if accuracy > best_acc:
            best_acc = accuracy
            checkpoint_path = save_dir / f'teacher_model_acc_{best_acc:.2f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, checkpoint_path)
            print(f"Saved teacher checkpoint to {checkpoint_path}")
    
    return model, best_acc