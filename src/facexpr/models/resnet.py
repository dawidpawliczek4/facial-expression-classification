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
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        return self.backbone(x)
