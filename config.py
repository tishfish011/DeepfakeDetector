"""
Configuration file for deepfake detection training and models
"""

import torch

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_CLASSES = 2

# Training configuration
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10  # Early stopping patience

# Model configurations
MODEL_CONFIGS = {
    'simple_cnn': {
        'name': 'Simple CNN',
        'description': 'Basic convolutional neural network',
        'channels': [64, 128, 256, 512]
    },
    'advanced_cnn': {
        'name': 'Advanced CNN with Attention',
        'description': 'Deep CNN with residual connections and spatial attention',
        'channels': [64, 128, 256, 512, 1024],
        'use_attention': True
    },
    'vision_transformer': {
        'name': 'Vision Transformer (ViT)',
        'description': 'Transformer-based model for image classification',
        'patch_size': 16,
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'mlp_ratio': 4
    },
    'cnn_transformer': {
        'name': 'CNN-Transformer Hybrid',
        'description': 'Combines CNN feature extraction with transformer attention',
        'cnn_channels': [64, 128, 256],
        'embed_dim': 512,
        'num_heads': 8,
        'num_layers': 6
    }
}

# Optimizer configuration
OPTIMIZER_CONFIG = {
    'name': 'AdamW',
    'lr': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'betas': (0.9, 0.999)
}

# Learning rate scheduler configuration
SCHEDULER_CONFIG = {
    'name': 'CosineAnnealingWarmRestarts',
    'T_0': 10,
    'T_mult': 2,
    'eta_min': 1e-6
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    'horizontal_flip_prob': 0.5,
    'rotation_degrees': 15,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'affine': {
        'degrees': 0,
        'translate': (0.1, 0.1),
        'scale': (0.9, 1.1)
    }
}

# Dataset configuration
DATASET_CONFIG = {
    'name': 'ComplexDataLab/OpenFake',
    'cache_dir': './data/cache',
    'train_max_samples': None,  # None for full dataset, or set a number for quick testing
    'test_max_samples': None
}

# Checkpoint configuration
CHECKPOINT_DIR = './checkpoints'
SAVE_BEST_ONLY = True
SAVE_FREQUENCY = 5  # Save every N epochs

# Logging configuration
LOG_DIR = './logs'
LOG_INTERVAL = 100  # Log every N batches

# Evaluation configuration
EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
CONFUSION_MATRIX_DIR = './results/confusion_matrices'
ROC_CURVE_DIR = './results/roc_curves'
