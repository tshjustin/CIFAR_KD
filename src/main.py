import os
import yaml
import json
import torch
import warnings
import argparse
from pathlib import Path
from models.CNN import CNN
import torch.optim as optim
from train import train, evaluate
from data.CIFAR10 import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from teacher_training import create_teacher_model, train_teacher

warnings.filterwarnings("ignore")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_params(params_path):
    with open(params_path, 'r') as f:
        return json.load(f)

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    project_root = Path(__file__).parent.parent
    data_dir = project_root / Path(config['data']['data_dir'])
    train_files = [str(data_dir / f"data_batch_{i}") for i in range(1, 6)]
    test_file = [str(data_dir / "test_batch")]

    train_dataset = CIFAR10(train_files, transform=transform)
    test_dataset = CIFAR10(test_file, transform=transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['train_batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['test_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    return train_loader, test_loader

def main(args):
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders(config)

    if args.train_teacher:
        print("Training teacher model...")
        teacher_model = create_teacher_model()
        teacher_model = teacher_model.to(device)
        teacher_model, teacher_acc = train_teacher(
            model=teacher_model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            device=device
        )
        print(f"Teacher model training completed with best accuracy: {teacher_acc:.2f}%")
        return

    student_model = CNN()
    student_model = student_model.to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=config['training']['learning_rate'])

    save_dir = project_root / config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        train(
            student_model=student_model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=config['training']['temperature'],
            alpha=config['training']['alpha'],
            log_interval=config['training']['log_interval'],
            teacher=config['training']['teacher']
        )
        
        accuracy = evaluate(
            model=student_model,
            test_loader=test_loader,
            device=device
        )
        
        if accuracy > best_acc:
            best_acc = accuracy
            checkpoint_path = save_dir / f'best_model_acc_{best_acc:.2f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 KD Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
    )
    parser.add_argument(
        "--train_teacher",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)