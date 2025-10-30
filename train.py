"""
Training script for deepfake detection models
Supports multiple architectures with comprehensive training pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.deepfake_detector import SimpleCNN
from models.advanced_cnn import AdvancedCNN
from models.vision_transformer import VisionTransformer
from models.cnn_transformer import CNNTransformerHybrid, CNNTransformerWithCLS
from data.openfake_dataset import get_dataloaders
import config


def get_model(model_name, num_classes=2):
    """Get model by name"""
    if model_name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name == 'advanced_cnn':
        cfg = config.MODEL_CONFIGS['advanced_cnn']
        return AdvancedCNN(
            num_classes=num_classes,
            channels=cfg['channels'],
            use_attention=cfg['use_attention']
        )
    elif model_name == 'vision_transformer':
        cfg = config.MODEL_CONFIGS['vision_transformer']
        return VisionTransformer(
            img_size=config.IMAGE_SIZE,
            patch_size=cfg['patch_size'],
            num_classes=num_classes,
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            num_layers=cfg['num_layers'],
            mlp_ratio=cfg['mlp_ratio']
        )
    elif model_name == 'cnn_transformer':
        cfg = config.MODEL_CONFIGS['cnn_transformer']
        return CNNTransformerHybrid(
            num_classes=num_classes,
            cnn_channels=cfg['cnn_channels'],
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            num_layers=cfg['num_layers']
        )
    elif model_name == 'cnn_transformer_cls':
        cfg = config.MODEL_CONFIGS['cnn_transformer']
        return CNNTransformerWithCLS(
            num_classes=num_classes,
            cnn_channels=cfg['cnn_channels'],
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            num_layers=cfg['num_layers']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def train(args):
    """Main training function"""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Get dataloaders
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        image_size=config.IMAGE_SIZE,
        num_workers=args.num_workers,
        train_max_samples=args.train_max_samples,
        test_max_samples=args.test_max_samples,
        cache_dir=config.DATASET_CONFIG['cache_dir']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.SCHEDULER_CONFIG['T_0'],
        T_mult=config.SCHEDULER_CONFIG['T_mult'],
        eta_min=config.SCHEDULER_CONFIG['eta_min']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 80)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'{args.model}_best.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
            print(f"New best model! Val Acc: {val_acc:.2f}%")
        
        # Save periodic checkpoint
        if epoch % args.save_frequency == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'{args.model}_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered after epoch {epoch}")
            break
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='advanced_cnn',
                       choices=['simple_cnn', 'advanced_cnn', 'vision_transformer', 
                               'cnn_transformer', 'cnn_transformer_cls'],
                       help='Model architecture to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY,
                       help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=config.PATIENCE,
                       help='Early stopping patience')
    parser.add_argument('--save-frequency', type=int, default=config.SAVE_FREQUENCY,
                       help='Save checkpoint every N epochs')
    
    # Data arguments
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS,
                       help='Number of data loading workers')
    parser.add_argument('--train-max-samples', type=int, default=None,
                       help='Maximum training samples (for quick testing)')
    parser.add_argument('--test-max-samples', type=int, default=None,
                       help='Maximum test samples (for quick testing)')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
