import pickle
import numpy as np 
import torch
import torchvision
from torch.utils.data import Dataset 
from typing import List, Optional, Callable, Tuple 
from pathlib import Path

def download_cifar10(root_dir: str):
    torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True)
    torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True)

def unpickle(file: str):
    file_path = Path(file)
    if not file_path.exists():
        download_cifar10(str(file_path.parent.parent))
    
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

class CIFAR10(Dataset):
    def __init__(self, annotation_files: List[str], transform: Optional[Callable] = None) -> None:
        self.images = []
        self.labels = []
        self.transform = transform

        first_file = Path(annotation_files[0])
        if not first_file.exists():
            download_cifar10(str(first_file.parent.parent))

        for file in annotation_files:
            data_dict = unpickle(file)
            self.images.append(data_dict['data'])   
            self.labels.extend(data_dict['labels'])

        self.images = np.concatenate(self.images, axis=0)

        # Reshape to (N, 3, 32, 32) and normalize to [0, 1], (-1) => Ensures correct spaces whilst maintaining correct number of dimensions 
        self.images = self.images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        # Get image and convert to torch tensor
        image = torch.from_numpy(self.images[idx]) # from_numpy => numpyArray => Tensor | Generally tensors are better for DL tasks 
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label