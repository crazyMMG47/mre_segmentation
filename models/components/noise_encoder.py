# simple cnn for mre noise encoder
import torch
import torch.nn as nn

class NoiseEncoder(nn.Module):
    """
    Encodes noise profiles into compact feature vectors.
    Processes 2D noise slices to extract uncertainty patterns.
    """
    def __init__(
        self,
        input_channels=1,
        base_channels=16,
        latent_dim=128,
        noise_size=(128, 128)
    ):
        super(NoiseEncoder, self).__init__()
        
        # Noise-specific feature extraction
        # Use smaller receptive fields to capture fine-grained patterns
        self.encoder = nn.Sequential(
            # Layer 1: Capture high-frequency patterns
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            
            # Layer 2: Combine local patterns
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            
            # Layer 3: Abstract noise patterns
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            
            # Layer 4: Compact representation
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to fixed size
        )
        
        # Compress to latent dimension
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8 * 4 * 4, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, noise):
        """
        Args:
            noise: (B, 1, H, W) - normalized noise profiles
        Returns:
            noise_features: (B, latent_dim) - compact noise features
        """
        x = self.encoder(noise)
        noise_features = self.fc(x)
        return noise_features