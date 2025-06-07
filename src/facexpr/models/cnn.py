import torch
import torch.nn as nn

class SimpleCnnModel(nn.Module):
    """
    A simple convolutional neural network for facial expression classification.

    Architecture:
        - 3 convolutional blocks (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d)
        - Flatten
        - Fully connected layers with dropout

    Args:
        num_classes (int): Number of output emotion classes.
        dropout_rate (float): Dropout probability between fully connected layers.
    """
    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.5):
        super(SimpleCnnModel, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 1x48x48 -> 32x48x48 -> 32x24x24
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 32x24x24 -> 64x24x24 -> 64x12x12
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 64x12x12 -> 128x12x12 -> 128x6x6
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Classifier for 128*6*6 -> FC layers -> num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, 1, 48, 48)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
