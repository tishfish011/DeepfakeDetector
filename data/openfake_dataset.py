import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict


class OpenFakeDataset(Dataset):
    """
    OpenFake dataset loader for deepfake detection
    Downloads and processes the ComplexDataLab/OpenFake dataset from Hugging Face
    """
    
    def __init__(
        self,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize OpenFake dataset
        
        Args:
            split: Dataset split ('train' or 'test')
            transform: Image transformations to apply
            max_samples: Maximum number of samples to load (None for all)
            cache_dir: Directory to cache downloaded dataset
        
        Note:
            This implementation loads the full dataset into memory.
            Use max_samples to limit memory usage during development/testing.
        """
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        
        # Load dataset (non-streaming for map-style access)
        print(f"Loading OpenFake dataset ({split} split)...")
        self.dataset = load_dataset(
            "ComplexDataLab/OpenFake",
            split=split,
            streaming=False,
            cache_dir=cache_dir
        )
        
        # Limit dataset if max_samples specified
        if max_samples is not None:
            total_samples = len(self.dataset)
            limit = min(max_samples, total_samples)
            self.dataset = self.dataset.select(range(limit))
            print(f"Dataset loaded: {limit} / {total_samples} samples (limited)")
        else:
            print(f"Dataset loaded: {len(self.dataset)} samples")
    
    def __len__(self):
        """Return dataset length"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (image_tensor, label, metadata)
                - image_tensor: Preprocessed image
                - label: 0 for real, 1 for fake
                - metadata: Dictionary with prompt and model info
        """
        sample = self.dataset[idx]
        
        # Get image
        image = sample['image']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get label (convert 'real'/'fake' to 0/1)
        label_str = sample['label']
        label = 0 if label_str == 'real' else 1
        
        # Get metadata
        metadata = {
            'prompt': sample.get('prompt', ''),
            'model': sample.get('model', 'unknown')
        }
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)
        
        return image, label, metadata


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training data augmentation transforms
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation)
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms for validation/testing
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_dataloaders(
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    train_max_samples: Optional[int] = None,
    test_max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for OpenFake dataset
    
    Args:
        batch_size: Batch size for dataloaders
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        train_max_samples: Maximum training samples (None for all)
        test_max_samples: Maximum test samples (None for all)
        cache_dir: Directory to cache dataset
        
    Returns:
        tuple: (train_loader, test_loader)
    
    Note:
        Use train_max_samples and test_max_samples to limit dataset size
        during development to reduce memory usage.
    """
    # Create datasets
    train_dataset = OpenFakeDataset(
        split='train',
        transform=get_train_transforms(image_size),
        max_samples=train_max_samples,
        cache_dir=cache_dir
    )
    
    test_dataset = OpenFakeDataset(
        split='test',
        transform=get_val_transforms(image_size),
        max_samples=test_max_samples,
        cache_dir=cache_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def collate_fn(batch):
    """
    Custom collate function for dataloader
    Handles variable-sized images and metadata
    """
    images = []
    labels = []
    metadata = []
    
    for image, label, meta in batch:
        images.append(image)
        labels.append(label)
        metadata.append(meta)
    
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels, metadata
