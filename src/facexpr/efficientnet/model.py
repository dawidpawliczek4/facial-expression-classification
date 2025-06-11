import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNetV2-based classifier for facial expression recognition.

    Args:
        num_classes (int): Number of target emotion classes.
    """

    def __init__(self, num_classes: int = 7, dropout=0.3):
        super(EfficientNetV2Classifier, self).__init__()

        self.backbone = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT)
        in_feats = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_feats, in_feats // 2),
            nn.BatchNorm1d(in_feats // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_feats // 2, num_classes)
        )

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        # Pass through our custom classifier
        output = self.classifier(features)
        return output
