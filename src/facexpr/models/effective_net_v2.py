import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNetV2-based classifier for facial expression recognition.

    Args:
        num_classes (int): Number of target emotion classes.
    """
    def __init__(self, num_classes: int = 7):
        super(EfficientNetV2Classifier, self).__init__()
        
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        
        # Get the number of features from the backbone
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with our own
        self.backbone.classifier = nn.Identity()
        
        # Create a new classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

def forward(self, x):
    # Extract features using the backbone
    features = self.backbone(x)
    # Pass through our custom classifier
    output = self.classifier(features)
    return output