import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# class ArcMarginProduct(nn.Module):
#     def __init__(self, in_features, out_features, s=30.0, m=0.50):
#         super().__init__()
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#         self.s, self.m = s, m

#     def forward(self, x, labels=None):
#         # Normalize feature i weights
#         cosine = F.linear(F.normalize(x), F.normalize(self.weight))
#         # kątowanie
#         if labels is None:
#             return cosine * self.s
#         theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
#         # tylko dla prawdziwych klas dodaj margines
#         target_logits = torch.cos(theta + self.m)
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, labels.view(-1,1), 1.0)
#         # skaluj i miksuj
#         output = self.s * (one_hot * target_logits + (1.0 - one_hot) * cosine)
#         return output


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel_size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Applies channel attention followed by spatial attention.
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel attention
        x = x * self.channel_att(x)
        # Spatial attention
        x = x * self.spatial_att(x)
        return x


class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNetV2-based classifier for facial expression recognition.

    Args:
        num_classes (int): Number of target emotion classes.
    """

    def __init__(self, num_classes: int = 7, dropout=0.2):
        super(EfficientNetV2Classifier, self).__init__()

        backbone = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT)        
        in_feats = backbone.classifier[1].in_features  # 1280

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.cbam = CBAM(in_planes=in_feats, ratio=16, kernel_size=7)
        self.pre_mlp = nn.Sequential(
            nn.BatchNorm1d(in_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_feats, in_feats // 2),
            nn.BatchNorm1d(in_feats // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # self.arc = ArcMarginProduct(in_feats // 2, num_classes, s=arc_s, m=arc_m)
        self.fc = nn.Linear(in_feats // 2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pre_mlp(x)        
        logits = self.fc(x)
        return logits
