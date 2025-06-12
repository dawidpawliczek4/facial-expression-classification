import torch.nn as nn
import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class SelfAttention2d(nn.Module):
    """
    Non-local self-attention blok 2D:
    - query/key/value poprzez 1×1 conv
    - attention map między wszystkimi pozycjami przestrzennymi
    - residual z uczonym skalarem gamma
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim,       kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        # projekcja
        proj_q = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B×(H*W)×C'
        proj_k = self.key_conv(x).view(B, -1, H * W)                     # B×C'×(H*W)
        # mapa uwagi
        energy = torch.bmm(proj_q, proj_k)                              # B×(H*W)×(H*W)
        attn  = self.softmax(energy)                                    # B×N×N
        # wartość
        proj_v = self.value_conv(x).view(B, -1, H * W)                  # B×C×N
        out = torch.bmm(proj_v, attn.permute(0, 2, 1))                  # B×C×N
        out = out.view(B, C, H, W)
        # dodajemy residual
        return self.gamma * out + x


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
        # self._init_identity()

    def _init_identity(self):
        # ustaw wagi = 0, bias = 0  → sigmoid(0)=0.5
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x * self.channel_att(x)
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

        #inject CBAM into backbone
        new_features = []
        stage_last = { 3: 64, 4:128, 5:160}
        for i, block in enumerate(backbone.features):
            new_features.append(block)
            if i in stage_last:
                new_features.append(CBAM(stage_last[i], ratio=16))
                new_features.append(SelfAttention2d(stage_last[i]))

        self.features = nn.Sequential(*new_features)

        self.cbam = CBAM(in_planes=in_feats, ratio=16, kernel_size=7)
        self.self_att = SelfAttention2d(in_dim=in_feats)

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

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        # with torch.amp.autocast(enabled=False):
        x = self.self_att(x)
        x = self.avgpool(x)        
        x = torch.flatten(x, 1)
        x = self.pre_mlp(x)        
        logits = self.fc(x)
        return logits