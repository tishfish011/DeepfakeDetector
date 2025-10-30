# Deepfake Detection System

Advanced AI-powered deepfake detection using CNN and Transformer models trained on the OpenFake dataset.

## Overview

This system provides state-of-the-art deepfake detection capabilities using multiple advanced architectures:
- **Simple CNN**: Basic convolutional neural network baseline
- **Advanced CNN with Attention**: Deep CNN with residual connections and spatial/channel attention mechanisms
- **Vision Transformer (ViT)**: Pure transformer-based architecture for image classification
- **CNN-Transformer Hybrid**: Combines CNN feature extraction with transformer attention
- **CNN-Transformer with CLS**: Hybrid model using classification token approach

## Dataset

The models are trained on the **OpenFake dataset** from Hugging Face (`ComplexDataLab/OpenFake`), which includes:
- **~3 million real images** from LAION-400M
- **~963,000 synthetic (fake) images** generated using state-of-the-art diffusion models
- Diverse content: faces, political figures, events, protests, disasters, memes
- High-quality deepfakes created with modern generative models

This dataset goes beyond traditional face-swap benchmarks, providing robust training for real-world deepfake detection.

## Features

- **Multiple Model Architectures**: Choose from 5 different model architectures
- **Image Detection**: Upload and analyze individual images
- **Video Detection**: Extract frames from videos and analyze them
- **Real-time Inference**: Fast prediction with confidence scores
- **Model Comparison**: Train and evaluate multiple architectures
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, AUC-ROC
- **Visualization**: Confusion matrices and ROC curves

## Installation

The required dependencies are already configured in this Replit environment:
- Python 3.x
- PyTorch
- Transformers
- Datasets (Hugging Face)
- Streamlit
- scikit-learn
- matplotlib, seaborn
- timm, tqdm

## Usage

### Running the Web Application

The Streamlit app is configured to run automatically:

```bash
streamlit run app.py --server.port 5000
```

The app provides:
1. **Model Selection**: Choose which architecture to use
2. **Detection Modes**:
   - Image Detection: Upload and analyze images
   - Video Detection: Analyze video frames
   - Sample Testing: Test with sample images

### Training Models

To train a model on the OpenFake dataset:

```bash
# Train Advanced CNN (recommended)
python train.py --model advanced_cnn --epochs 50 --batch-size 32

# Train Vision Transformer
python train.py --model vision_transformer --epochs 50 --batch-size 16

# Train CNN-Transformer Hybrid
python train.py --model cnn_transformer --epochs 50 --batch-size 32

# Quick test with limited samples
python train.py --model advanced_cnn --epochs 5 --train-max-samples 10000 --test-max-samples 2000
```

**Training Arguments**:
- `--model`: Model architecture (simple_cnn, advanced_cnn, vision_transformer, cnn_transformer, cnn_transformer_cls)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Initial learning rate (default: 1e-4)
- `--weight-decay`: Weight decay for optimizer (default: 1e-5)
- `--patience`: Early stopping patience (default: 10)
- `--train-max-samples`: Limit training samples for testing (default: None = full dataset)
- `--test-max-samples`: Limit test samples for testing (default: None = full dataset)

**Note**: Training on the full OpenFake dataset requires significant resources (GPU recommended). Use `--train-max-samples` and `--test-max-samples` for quick testing.

### Evaluating Models

To evaluate a trained model:

```bash
# Evaluate Advanced CNN
python evaluate.py --model advanced_cnn

# Evaluate with specific checkpoint
python evaluate.py --model vision_transformer --checkpoint ./checkpoints/vision_transformer_epoch_30.pth

# Quick evaluation with limited samples
python evaluate.py --model advanced_cnn --test-max-samples 5000
```

**Evaluation Outputs**:
- Accuracy, Precision, Recall, F1 Score, AUC-ROC
- Confusion matrix (saved to `results/confusion_matrices/`)
- ROC curve (saved to `results/roc_curves/`)
- Detailed classification report
- Metrics JSON file (saved to `results/`)

### Model Checkpoints

Trained models are saved to `./checkpoints/`:
- `{model_name}_best.pth`: Best model based on validation accuracy
- `{model_name}_epoch_{N}.pth`: Periodic checkpoints every N epochs

The web app automatically loads the best checkpoint if available.

## Model Architectures

### 1. Simple CNN
- Basic 4-layer convolutional network
- ~2M parameters
- Good baseline performance
- Fast inference

### 2. Advanced CNN with Attention
- Deep residual architecture with 5 stages
- Spatial and channel attention mechanisms
- ~25M parameters
- Superior feature extraction
- Recommended for best accuracy

### 3. Vision Transformer (ViT)
- Pure transformer architecture
- Patch-based image processing
- 12 transformer layers, 12 attention heads
- ~86M parameters
- Excellent for capturing global patterns
- Requires more training data

### 4. CNN-Transformer Hybrid
- CNN backbone for local features
- Transformer encoder for global attention
- ~15M parameters
- Balanced approach
- Good accuracy/efficiency trade-off

### 5. CNN-Transformer with CLS Token
- Similar to hybrid but uses classification token
- ViT-style classification
- ~15M parameters
- Alternative hybrid approach

## Configuration

Edit `config.py` to customize:
- Image size and batch size
- Training hyperparameters
- Model configurations
- Dataset paths
- Checkpoint and logging settings

## Directory Structure

```
.
├── app.py                      # Streamlit web application
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── config.py                   # Configuration file
├── models/
│   ├── deepfake_detector.py   # Main detector class
│   ├── advanced_cnn.py        # Advanced CNN architecture
│   ├── vision_transformer.py  # Vision Transformer
│   └── cnn_transformer.py     # CNN-Transformer hybrid
├── data/
│   ├── openfake_dataset.py    # OpenFake dataset loader
│   └── cache/                 # Dataset cache (auto-created)
├── utils/
│   ├── image_processor.py     # Image utilities
│   └── video_processor.py     # Video utilities
├── checkpoints/               # Trained model checkpoints
├── results/                   # Evaluation results
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── *.json                # Metrics files
└── README.md                  # This file
```

## Performance Expectations

With proper training on the full OpenFake dataset:
- **Advanced CNN**: 85-92% accuracy
- **Vision Transformer**: 87-94% accuracy
- **CNN-Transformer Hybrid**: 86-93% accuracy

*Note: Actual performance depends on training duration, hyperparameters, and computational resources.*

## Dataset Details

The OpenFake dataset (`ComplexDataLab/OpenFake`) provides:
- **Balanced splits**: Train/test split with ~60k test images
- **Diverse generators**: Images from multiple state-of-the-art models
- **Rich metadata**: Prompts, generator model names
- **High quality**: ~1 megapixel images with varied aspect ratios
- **Real-world relevance**: Politically-themed content, events, faces

Citation:
```
OpenFake: An Open Dataset and Platform Toward Large-Scale Deepfake Detection
Victor Livernoche, Akshatha Arodi, Andreea Musulan, et al.
arXiv:2509.09495, September 2025
```

## Tips for Best Results

1. **Start with a subset**: Use `--train-max-samples 10000` for quick experimentation
2. **Use GPU**: Training is much faster on GPU (CUDA)
3. **Monitor training**: Watch for overfitting; early stopping is enabled by default
4. **Data augmentation**: Already included in training pipeline
5. **Model selection**: Advanced CNN offers best accuracy/efficiency trade-off
6. **Checkpoint selection**: Use best checkpoint for inference, not latest

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Use smaller model (simple_cnn or cnn_transformer)
- Limit dataset with `--train-max-samples`

### Slow Training
- Reduce `--num-workers` if CPU bottleneck
- Use smaller image size in config.py
- Enable GPU/CUDA if available

### Poor Accuracy
- Train for more epochs
- Check dataset is loading correctly
- Try different model architecture
- Verify data augmentation is working

## Future Enhancements

- [ ] Multi-scale detection
- [ ] Ensemble methods
- [ ] GradCAM visualization
- [ ] Additional datasets
- [ ] Model quantization
- [ ] ONNX export for deployment

## License

This project is for educational and research purposes. The OpenFake dataset has CC-BY-SA-4.0 license with non-commercial restrictions for some subsets.

## Acknowledgments

- OpenFake dataset: ComplexDataLab
- Hugging Face for dataset hosting
- PyTorch and Transformers libraries
- Streamlit for web interface
