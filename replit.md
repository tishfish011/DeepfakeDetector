# Deepfake Detection System

## Overview

An advanced AI-powered deepfake detection system that leverages multiple deep learning architectures to identify synthetic/manipulated images and videos. The system is trained on the OpenFake dataset from Hugging Face (ComplexDataLab/OpenFake), which contains ~3 million real images and ~963,000 synthetic images generated using state-of-the-art diffusion models. The application provides a Streamlit web interface for uploading and analyzing both images and videos, with real-time inference and comprehensive evaluation metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Decision: Streamlit Web Application**
- **Rationale**: Provides rapid prototyping with minimal frontend code while offering interactive widgets for file uploads, model selection, and result visualization
- **Components**: 
  - Image upload and analysis interface
  - Video frame extraction and batch processing
  - Real-time confidence score display
  - Model selection dropdown for switching between architectures
- **Session State Management**: Uses Streamlit's session state to persist loaded models and prevent redundant model loading

### Backend Architecture

**Decision: PyTorch-based Multi-Model Architecture**
- **Problem**: Need to support multiple detection approaches with varying complexity and accuracy tradeoffs
- **Solution**: Implemented 5 distinct model architectures as separate modules:
  1. **SimpleCNN**: Baseline 4-layer convolutional network with batch normalization
  2. **AdvancedCNN**: Deep CNN with residual connections, spatial attention, and channel attention mechanisms
  3. **VisionTransformer**: Pure transformer-based architecture using patch embeddings (ViT)
  4. **CNNTransformerHybrid**: Combines CNN feature extraction backbone with transformer attention layers
  5. **CNNTransformerWithCLS**: Hybrid model using classification token approach similar to BERT
- **Alternatives Considered**: Single unified architecture with configurable components
- **Pros**: Modularity enables independent optimization, easy comparison, and flexibility to choose based on deployment constraints
- **Cons**: Increased codebase complexity and maintenance overhead

**Decision: Checkpoint-based Model Persistence**
- **Problem**: Need to save and load trained models efficiently
- **Solution**: Each model saves its best performing checkpoint to `./checkpoints/{model_name}_best.pth` during training
- **Structure**: Checkpoints include model state dict, optimizer state, epoch number, and best validation accuracy

**Decision: Configuration-Driven Design**
- **Rationale**: Centralized `config.py` manages hyperparameters, model architectures, and training settings
- **Benefits**: Easy experimentation with different configurations without code changes
- **Key Parameters**: Image size (224x224), batch size (32), learning rate (1e-4), early stopping patience (10 epochs)

### Data Architecture

**Decision: HuggingFace Datasets Integration**
- **Problem**: Need to efficiently load and process large-scale OpenFake dataset (~4M images)
- **Solution**: Direct integration with HuggingFace `datasets` library for streaming and caching
- **Implementation**: Custom `OpenFakeDataset` class wraps HuggingFace dataset with PyTorch Dataset interface
- **Memory Management**: Supports `max_samples` parameter to limit memory usage during development/testing
- **Caching**: Leverages HuggingFace's built-in caching to avoid re-downloading

**Decision: PyTorch DataLoader with Transformations**
- **Preprocessing Pipeline**: 
  - Resize to 224x224
  - Random horizontal flips (training augmentation)
  - Normalization using ImageNet statistics
  - ToTensor conversion
- **Batching**: Configurable batch size with multi-worker data loading for performance

### Training Architecture

**Decision: Comprehensive Training Pipeline with Early Stopping**
- **Components**:
  - Cross-entropy loss function
  - Adam optimizer with weight decay (1e-5)
  - Cosine annealing learning rate scheduler with warm restarts
  - Early stopping based on validation loss (patience: 10 epochs)
  - Best model checkpointing
- **Rationale**: Prevents overfitting on large dataset while maximizing performance

**Decision: Multi-Metric Evaluation Framework**
- **Metrics Tracked**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Visualization**: Confusion matrices and ROC curves generated during evaluation
- **Use Case**: Enables comprehensive model comparison beyond single accuracy metric

### Attention Mechanisms

**Decision: Dual Attention in Advanced CNN**
- **Problem**: CNNs may not focus on manipulation artifacts
- **Solution**: Implemented both spatial and channel attention
  - **Spatial Attention**: Uses max/avg pooling to generate attention maps highlighting important image regions
  - **Channel Attention**: Weights feature channels based on global pooling and learned transformations
- **Inspiration**: CBAM (Convolutional Block Attention Module)

**Decision: Multi-Head Self-Attention in Transformers**
- **Implementation**: Standard scaled dot-product attention with multiple heads
- **Parameters**: 12 heads for ViT, 8 heads for hybrid models
- **Positional Encoding**: Learnable 2D positional embeddings for spatial feature maps

### Image and Video Processing

**Decision: Modular Processor Classes**
- **ImageProcessor**: Handles format validation, loading, resizing, and preprocessing
- **VideoProcessor**: Extracts frames from videos (requires OpenCV - noted in code but not currently installed)
- **Supported Formats**: 
  - Images: JPG, PNG, BMP, TIFF, WebP
  - Videos: MP4, AVI, MOV, MKV, WMV, FLV

**Decision: PIL-based Image Handling**
- **Rationale**: Seamless integration with PyTorch transforms and cross-platform compatibility
- **Auto-conversion**: Automatically converts images to RGB mode

## External Dependencies

### Machine Learning Frameworks
- **PyTorch**: Core deep learning framework for model implementation, training, and inference
- **Transformers (HuggingFace)**: Provides transformer components and pre-trained model utilities
- **timm**: PyTorch Image Models library for advanced computer vision architectures

### Dataset and Data Processing
- **datasets (HuggingFace)**: Loading and streaming the OpenFake dataset from HuggingFace Hub
- **Pillow (PIL)**: Image loading, preprocessing, and format conversion
- **NumPy**: Numerical operations and array manipulations

### Web Application
- **Streamlit**: Interactive web interface for model deployment and inference
- **Configuration**: Auto-runs on Replit environment

### Evaluation and Visualization
- **scikit-learn**: Metrics calculation (accuracy, precision, recall, F1, confusion matrix, ROC curves)
- **matplotlib**: Plotting confusion matrices and performance graphs
- **seaborn**: Enhanced visualization styling
- **tqdm**: Progress bars for training and evaluation loops

### Dataset Source
- **ComplexDataLab/OpenFake**: HuggingFace dataset containing 3M real images (LAION-400M) and 963K synthetic images from modern diffusion models
- **Access**: Publicly available through HuggingFace Datasets Hub

### Video Processing (Optional)
- **OpenCV (cv2)**: Video frame extraction and processing (noted in code but not currently installed in environment)
- **Status**: VideoProcessor class prepared for integration when OpenCV is added

### Hardware Acceleration
- **CUDA**: Automatic GPU utilization when available via PyTorch
- **Fallback**: CPU-based training and inference supported