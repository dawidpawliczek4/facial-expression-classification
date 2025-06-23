import torch
import torch.nn as nn
from torchvision import models
from facexpr.utils.attention import CBAM, MultiHeadSelfAttention2d

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout=0.2):
        super(ResNetClassifier, self).__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )
        self.cbam = CBAM(512)
        self.attn = MultiHeadSelfAttention2d(in_dim=512, heads=8)
        in_feats=512
        self.pre_mlp = nn.Sequential(
            nn.BatchNorm1d(in_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_feats, in_feats // 2),
            nn.BatchNorm1d(in_feats // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(in_feats // 2, num_classes)

    
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.attn(x)
        x = torch.flatten(x, 1)
        x = self.pre_mlp(x)
        logits = self.fc(x)
        return logits