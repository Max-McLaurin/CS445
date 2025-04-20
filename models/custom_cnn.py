# models/custom_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    """
    A simple CNN feature extractor expecting 6-channel input (concatenated pair of RGB frames).
    """
    def __init__(self, in_channels=6, feature_dim=512):
        super(CustomCNN, self).__init__()
        # First convolution: input channels = 6, output = 32 feature maps
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm2d(32)
        # Second convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm2d(64)
        # Third convolution
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        # Fourth convolution to feature_dim channels
        self.conv4 = nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(feature_dim)

    def forward(self, x):
        """
        Forward pass of the CNN.
        Args:
            x: Tensor of shape (B, 6, H, W)
        Returns:
            Tensor of shape (B, feature_dim, H_out, W_out)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x


