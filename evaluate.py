"""
Evaluation script for deepfake detection models
Provides comprehensive metrics, confusion matrices, and ROC curves
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import os
import argparse
from tqdm import tqdm
import json

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


def load_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint val_acc: {checkpoint['val_acc']:.2f}%")
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and labels"""
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary'),
    }
    
    # Calculate AUC-ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    metrics['auc_roc'] = auc(fpr, tpr)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to {save_path}")


def evaluate(args):
    """Main evaluation function"""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.CONFUSION_MATRIX_DIR, exist_ok=True)
    os.makedirs(config.ROC_CURVE_DIR, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load test data
    print("Loading test dataset...")
    _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        image_size=config.IMAGE_SIZE,
        num_workers=args.num_workers,
        test_max_samples=args.test_max_samples,
        cache_dir=config.DATASET_CONFIG['cache_dir']
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint or os.path.join(
        config.CHECKPOINT_DIR,
        f'{args.model}_best.pth'
    )
    
    if os.path.exists(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path, device)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Evaluating with randomly initialized weights")
    
    # Evaluate
    print("\nEvaluating model...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Print results
    print("\n" + "="*60)
    print(f"Evaluation Results for {args.model}")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("="*60)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Real', 'Fake'],
                                digits=4))
    
    # Plot confusion matrix
    cm_path = os.path.join(config.CONFUSION_MATRIX_DIR, 
                          f'{args.model}_confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(config.ROC_CURVE_DIR, 
                           f'{args.model}_roc_curve.png')
    plot_roc_curve(y_true, y_prob, roc_path)
    
    # Save metrics to JSON
    results_path = os.path.join('results', f'{args.model}_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='advanced_cnn',
                       choices=['simple_cnn', 'advanced_cnn', 'vision_transformer', 
                               'cnn_transformer', 'cnn_transformer_cls'],
                       help='Model architecture to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: best checkpoint)')
    
    # Data arguments
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS,
                       help='Number of data loading workers')
    parser.add_argument('--test-max-samples', type=int, default=None,
                       help='Maximum test samples (for quick testing)')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()
