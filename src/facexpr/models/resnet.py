import torch.nn as nn
from torchvision import models

class ResNet50Classifier(nn.Module):
    """
    ResNet50-based classifier for facial expression recognition.

    Args:
        num_classes (int): Number of target emotion classes.
        pretrained (bool): Whether to load ImageNet-pretrained weights.
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super(ResNet50Classifier, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        
        in_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Create a new classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Pass through the classifier
        output = self.classifier(features)
        
        return output