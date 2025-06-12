import torch
import torch.nn as nn

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
        total_dim = in_dim  # heads * head_dim
        # projekcje Q, K, V
        self.query_conv = nn.Conv2d(in_dim, total_dim, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, total_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, total_dim, kernel_size=1)
        # output projection
        self.out_proj = nn.Conv2d(total_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.scale = (self.head_dim ** -0.5)

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        # projekcje
        q = self.query_conv(x).view(B, self.heads, self.head_dim, N)  # B×h×d×N
        k = self.key_conv(x).view(B, self.heads, self.head_dim, N)    # B×h×d×N
        v = self.value_conv(x).view(B, self.heads, self.head_dim, N)  # B×h×d×N
        # reshape do (B,h,N,d)
        q = q.permute(0, 1, 3, 2)  # B×h×N×d
        k = k  # B×h×d×N
        v = v.permute(0, 1, 3, 2)  # B×h×N×d
        # attention
        energy = torch.matmul(q, k) * self.scale  # B×h×N×N
        attn = self.softmax(energy)
        out = torch.matmul(attn, v)  # B×h×N×d
        # powrot do B×(h*d)×H×W
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        return self.gamma * out + x