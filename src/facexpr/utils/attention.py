import torch.nn as nn
import torch


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
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim,       kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, explain=False):
        B, C, H, W = x.size()

        proj_q = self.query_conv(x).view(
            B, -1, H * W).permute(0, 2, 1)
        proj_k = self.key_conv(x).view(
            B, -1, H * W)
        
        energy = torch.bmm(proj_q, proj_k)
        attn = self.softmax(energy)
        proj_v = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        y = self.gamma * out + x
        return (y, attn) if explain else y


class MultiHeadSelfAttention2d(nn.Module):
    """
    Multi-head self-attention 2D:
    - query/key/value przez 1×1 conv
    - H heads, każdy head uwagi działa niezależnie na spatial dim
    - output projection + residual z uczonym skalarem gamma
    """

    def __init__(self, in_dim, heads=8):
        super().__init__()
        assert in_dim % heads == 0, "in_dim must be divisible by number of heads"
        self.in_dim = in_dim
        self.heads = heads
        self.head_dim = in_dim // heads
        total_dim = in_dim

        self.query_conv = nn.Conv2d(in_dim, total_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, total_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, total_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(total_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.scale = (self.head_dim ** -0.5)

    def forward(self, x, explain=False):
        B, C, H, W = x.size()
        N = H * W
        q = self.query_conv(x).view(B, self.heads, self.head_dim, N)
        k = self.key_conv(x).view(B, self.heads, self.head_dim, N)
        v = self.value_conv(x).view(B, self.heads, self.head_dim, N)
        
        q = q.permute(0, 1, 3, 2)
        k = k
        v = v.permute(0, 1, 3, 2)
        
        energy = torch.matmul(q, k) * self.scale
        attn = self.softmax(energy)
        out = torch.matmul(attn, v)
        
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        y = self.gamma * out + x
        return (y, attn) if explain else y


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

    def forward(self, x, explain=False):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        cam = self.sigmoid(avg_out + max_out)
        out = x * cam
        return (out, cam) if explain else out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel_size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, explain=False):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        smap = self.sigmoid(self.conv(x_cat))
        out = x * smap
        return (out, smap) if explain else out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Applies channel attention followed by spatial attention.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x, explain=False):
        if explain:
            x_ca, ch_map = self.channel_att(x, explain=True)
            x_sa, sp_map = self.spatial_att(x_ca, explain=True)
            return x_sa, ch_map, sp_map
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
