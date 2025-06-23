import torch.nn as nn
import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from facexpr.utils.attention import CBAM, SelfAttention2d, MultiHeadSelfAttention2d

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

        new_features = []
        stage_last = {2: 48, 3: 64, 4: 128, 5: 160}
        for i, block in enumerate(backbone.features):
            new_features.append(block)
            if i in stage_last:
                new_features.append(CBAM(stage_last[i], ratio=16))
                new_features.append(SelfAttention2d(stage_last[i]))

        self.features = nn.Sequential(*new_features)

        self.cbam = CBAM(in_planes=in_feats, ratio=16, kernel_size=7)
        self.self_att = MultiHeadSelfAttention2d(in_dim=in_feats)

        self.avgpool = backbone.avgpool
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

    def forward(self, x, explain=False):
        x = self.features(x)
        if explain:
            x, ch_map, sp_map = self.cbam(x, explain=True)
            x, attn = self.self_att(x, explain=True)
        else:
            x = self.cbam(x)
            x = self.self_att(x)        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pre_mlp(x)
        logits = self.fc(x)
        if explain:
            return logits, ch_map, sp_map, attn
        return logits
