# src/model.py
"""Autoencoder model for anomaly detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAutoencoder(nn.Module):
    """
    Autoencoder for anomaly detection
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder - Convolutional layers
        self.encoder = nn.Sequential(
            # Input: 1 x 128 x 128
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Calculate flattened size: 256 * 8 * 8 = 16384
        self.flatten_size = 256 * 8 * 8
        
        # Bottleneck
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)  # Bottleneck: 64 dimensions
        
        # Decoder
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, self.flatten_size)
        
        # Decoder convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 1 x 128 x 128
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)  # Flatten
        
        # Bottleneck
        bottleneck = F.relu(self.fc1(encoded))
        latent = self.fc2(bottleneck)
        
        # Decode from bottleneck
        decoded = F.relu(self.fc3(latent))
        decoded = self.fc4(decoded)
        decoded = decoded.view(decoded.size(0), 256, 8, 8)  # Reshape
        
        # Decoder convolutions
        reconstructed = self.decoder(decoded)
        
        return reconstructed, latent
    
    def compute_anomaly_score(self, x):
        """Compute anomaly score for input"""
        with torch.no_grad():
            reconstructed, latent = self.forward(x)
            # Mean squared error
            score = F.mse_loss(reconstructed, x, reduction='mean')
        return score.item()