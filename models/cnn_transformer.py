"""
CNN-Transformer Hybrid Model
Combines CNN feature extraction with Transformer attention for deepfake detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """CNN-based feature extraction backbone"""
    
    def __init__(self, in_channels=3, channels=[64, 128, 256]):
        super(CNNFeatureExtractor, self).__init__()
        
        layers = []
        current_channels = in_channels
        
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, 
                         padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                         padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            current_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.out_channels = channels[-1]
    
    def forward(self, x):
        return self.features(x)


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial features"""
    
    def __init__(self, channels):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, channels, 1, 1))
    
    def forward(self, x):
        # Add positional encoding
        return x + self.pos_embed


class TransformerEncoder(nn.Module):
    """Transformer encoder for feature refinement"""
    
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, x):
        return self.transformer(x)


class CNNTransformerHybrid(nn.Module):
    """
    Hybrid CNN-Transformer model for deepfake detection
    Combines local feature extraction (CNN) with global attention (Transformer)
    """
    
    def __init__(self, num_classes=2, cnn_channels=[64, 128, 256],
                 embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4, dropout=0.1):
        super(CNNTransformerHybrid, self).__init__()
        
        # CNN feature extractor
        self.cnn_features = CNNFeatureExtractor(in_channels=3, 
                                                channels=cnn_channels)
        
        # Project CNN features to transformer embedding dimension
        self.feature_proj = nn.Conv2d(cnn_channels[-1], embed_dim, 
                                     kernel_size=1)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(embed_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract CNN features
        # Input: (B, 3, 224, 224)
        cnn_features = self.cnn_features(x)  # (B, C, H', W')
        
        # Project to embedding dimension
        features = self.feature_proj(cnn_features)  # (B, embed_dim, H', W')
        
        # Add positional encoding
        features = self.pos_encoding(features)  # (B, embed_dim, H', W')
        
        # Reshape for transformer: (B, H'*W', embed_dim)
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        
        # Apply transformer
        features = self.transformer(features)  # (B, H'*W', embed_dim)
        
        # Global average pooling
        features = features.mean(dim=1)  # (B, embed_dim)
        
        # Apply normalization
        features = self.norm(features)
        
        # Classification
        output = self.classifier(features)  # (B, num_classes)
        
        return output


class CNNTransformerWithCLS(nn.Module):
    """
    CNN-Transformer with CLS token (similar to ViT approach)
    """
    
    def __init__(self, num_classes=2, cnn_channels=[64, 128, 256],
                 embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4, dropout=0.1):
        super(CNNTransformerWithCLS, self).__init__()
        
        # CNN feature extractor
        self.cnn_features = CNNFeatureExtractor(in_channels=3, 
                                                channels=cnn_channels)
        
        # Project CNN features to transformer embedding dimension
        self.feature_proj = nn.Conv2d(cnn_channels[-1], embed_dim, 
                                     kernel_size=1)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(embed_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Extract CNN features
        cnn_features = self.cnn_features(x)  # (B, C, H', W')
        
        # Project to embedding dimension
        features = self.feature_proj(cnn_features)  # (B, embed_dim, H', W')
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Reshape for transformer
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)
        
        # Apply transformer
        features = self.transformer(features)
        
        # Use CLS token for classification
        cls_output = features[:, 0]
        cls_output = self.norm(cls_output)
        output = self.head(cls_output)
        
        return output
