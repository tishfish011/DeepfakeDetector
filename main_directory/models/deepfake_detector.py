import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

def get_checkpoint_path(model_name):
    """Get checkpoint path for a model if it exists"""
    checkpoint_path = "main_directory/deepfake_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"✅ Found checkpoint at: {checkpoint_path}")
        detector.load_state_dict(torch.load(checkpoint_path, map_location=device)) #Delete later if not needed or change device to cpu
        return checkpoint_path
    else:
        print(f"⚠️ Checkpoint not found at: {checkpoint_path}")
        return None

class SimpleCNN(nn.Module):
    """Simple CNN for deepfake detection"""
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DeepfakeDetector:
    def __init__(self, model_name='simple_cnn', checkpoint_path=None):
        """
        Initialize the deepfake detector with a model
        
        Args:
            model_name: Name of the model architecture
                Options: 'simple_cnn', 'advanced_cnn', 'vision_transformer', 
                         'cnn_transformer', 'cnn_transformer_cls'
            checkpoint_path: Path to trained model checkpoint (optional)
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.model_info = {}
        
        self._load_model()
        self._setup_transforms()
    
    def _load_model(self):
        """Load the model architecture and optionally load checkpoint"""
        try:
            # Import model classes
            from models.advanced_cnn import AdvancedCNN
            from models.vision_transformer import VisionTransformer
            from models.cnn_transformer import CNNTransformerHybrid, CNNTransformerWithCLS
            
            # Create model based on architecture
            if self.model_name == 'simple_cnn':
                self.model = SimpleCNN(num_classes=2)
                
            elif self.model_name == 'advanced_cnn':
                self.model = AdvancedCNN(
                    num_classes=2,
                    channels=[64, 128, 256, 512, 1024],
                    use_attention=True
                )
                
            elif self.model_name == 'vision_transformer':
                self.model = VisionTransformer(
                    img_size=224,
                    patch_size=16,
                    num_classes=2,
                    embed_dim=768,
                    num_heads=12,
                    num_layers=12,
                    mlp_ratio=4
                )
                
            elif self.model_name == 'cnn_transformer':
                self.model = CNNTransformerHybrid(
                    num_classes=2,
                    cnn_channels=[64, 128, 256],
                    embed_dim=512,
                    num_heads=8,
                    num_layers=6
                )
                
            elif self.model_name == 'cnn_transformer_cls':
                self.model = CNNTransformerWithCLS(
                    num_classes=2,
                    cnn_channels=[64, 128, 256],
                    embed_dim=512,
                    num_heads=8,
                    num_layers=6
                )
                
            #Chat GTP Addition
            elif self.model_name == "resnet18":
                from torchvision.models import resnet18
                self.model = resnet18(num_classes=2)
                
            else:
                raise ValueError(f"Unknown model architecture: {self.model_name}")
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Load checkpoint if provided
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                self._load_checkpoint()
                print(f"Loaded checkpoint from {self.checkpoint_path}")
            else:
                # Initialize weights for untrained model
                if self.model_name == 'simple_cnn':
                    self._initialize_weights()
                print(f"Using {'randomly initialized' if not self.checkpoint_path else 'untrained'} {self.model_name} model")
            
            # Set to evaluation mode
            self.model.eval()
            
            # Store model info
            self.model_info = {
                'architecture': self.model_name,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'trained': self.checkpoint_path is not None and os.path.exists(self.checkpoint_path) if self.checkpoint_path else False
            }
            
        except Exception as e:
            raise Exception(f"Failed to load model {self.model_name}: {str(e)}")
    
    '''
    def _load_checkpoint(self):
        """Load model weights from checkpoint"""
        try:
            #checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            #self.model.load_state_dict(checkpoint['model_state_dict'])
           
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Handle both dictionary and plain state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            
            # Store checkpoint info
            self.model_info['checkpoint_epoch'] = checkpoint.get('epoch', 'unknown')
            self.model_info['checkpoint_val_acc'] = checkpoint.get('val_acc', 'unknown')
            
        except Exception as e:
            raise Exception(f"Failed to load checkpoint from {self.checkpoint_path}: {str(e)}") '''

    def _load_checkpoint(self):
        """Load model weights from checkpoint"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

            # Handle both dictionary and plain state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            # Store checkpoint info (only if it's a dict)
            if isinstance(checkpoint, dict):
                self.model_info['checkpoint_epoch'] = checkpoint.get('epoch', 'unknown')
                self.model_info['checkpoint_val_acc'] = checkpoint.get('val_acc', 'unknown')
            else:
                self.model_info['checkpoint_epoch'] = 'N/A'
                self.model_info['checkpoint_val_acc'] = 'N/A'

        except Exception as e:
            raise Exception(f"Failed to load checkpoint from {self.checkpoint_path}: {str(e)}")

    
    def _initialize_weights(self):
        """Initialize model weights with Xavier initialization"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        # Normalize with ImageNet stats (standard practice)
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """Create transformation pipeline"""
        class ResizeTransform:
            def __call__(self, img):
                return img.resize((224, 224), Image.LANCZOS)
        
        class ToTensorTransform:
            def __call__(self, img):
                # Convert PIL Image to tensor
                img_array = np.array(img).astype(np.float32) / 255.0
                # Transpose from HWC to CHW
                img_array = np.transpose(img_array, (2, 0, 1))
                return torch.from_numpy(img_array)
        
        class NormalizeTransform:
            def __init__(self, mean, std):
                self.mean = torch.tensor(mean).view(-1, 1, 1)
                self.std = torch.tensor(std).view(-1, 1, 1)
            
            def __call__(self, tensor):
                return (tensor - self.mean) / self.std
        
        class Compose:
            def __init__(self, transforms):
                self.transforms = transforms
            
            def __call__(self, img):
                for t in self.transforms:
                    img = t(img)
                return img
        
        return Compose([
            ResizeTransform(),
            ToTensorTransform(),
            NormalizeTransform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                image = Image.fromarray(image.astype(np.uint8))
            else:
                # Assume RGB format
                image = Image.fromarray(image.astype(np.uint8))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict_image(self, image):
        """
        Predict if an image is real or deepfake
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            tuple: (prediction, confidence)
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Convert to CPU and numpy
                confidence = confidence.cpu().numpy()[0]
                predicted = predicted.cpu().numpy()[0]
                
                # Map prediction to label
                labels = ['Real', 'Deepfake']
                prediction = labels[predicted]
                
                return prediction, float(confidence)
                
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def predict_batch(self, images):
        """
        Predict multiple images at once
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            List of tuples: [(prediction, confidence), ...]
        """
        try:
            # Preprocess all images
            batch_tensors = []
            for image in images:
                tensor = self.preprocess_image(image)
                batch_tensors.append(tensor)
            
            # Stack tensors into batch
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, 1)
                
                # Convert to CPU and numpy
                confidences = confidences.cpu().numpy()
                predictions = predictions.cpu().numpy()
                
                # Map predictions to labels
                labels = ['Real', 'Deepfake']
                results = []
                for pred, conf in zip(predictions, confidences):
                    results.append((labels[pred], float(conf)))
                
                return results
                
        except Exception as e:
            raise Exception(f"Error during batch prediction: {str(e)}")
    
    def get_model_info(self):
        """Get information about the loaded model"""
        info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        info.update(self.model_info)
        return info
