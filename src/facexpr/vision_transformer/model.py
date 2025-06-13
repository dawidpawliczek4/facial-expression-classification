import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=config["in_channels"],
                out_channels=config["embed_dim"],
                kernel_size=config["patch_size"],
                stride=config["patch_size"]
            ),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.zeros(size=(1, config["in_channels"], config["embed_dim"])), requires_grad=True)
        self.positional_embedding = nn.Parameter(torch.randn(size=(1, config["num_patches"] + 1, config["embed_dim"])), requires_grad=True)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        B = x.size(0) # Rozmiar batchu
        cls_token = self.cls_token.expand(B, -1, -1) # Rozszerzenie tokena klasy do rozmiaru batchu
        x = self.patcher(x).transpose(1, 2) # (B, E, N) -> (B, N, E)
        x = torch.cat([cls_token, x], dim=1)
        x = self.positional_embedding + x
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout, activation, batch_first):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim)
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        attention_output, _ = self.mha(x, x, x)
        x = x + attention_output # Residual connection
        x = self.layer_norm(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output # Residual connection
        return x
        
    
class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding_block = PatchEmbedding(config)
        
        self.encoder_block = nn.ModuleList([
            TransformerEncoderLayer(
                config['embed_dim'],
                config['num_heads'],
                config['hidden_dim'],
                config['dropout'],
                config['activation'],
                True,
            )
            for _ in range(config['num_encoders'])
        ])
        
        self.classifier_block = nn.Sequential(
            nn.LayerNorm(normalized_shape=config['embed_dim']),
            nn.Linear(config['embed_dim'], config['num_classes'])
        )
    
    def forward(self, x):
        x = self.patch_embedding_block(x)
        for encoder in self.encoder_block:
            x = encoder(x)
        x = x[:, 0, :] # Pobranie tokena klasy (CLS token)
        x = self.classifier_block(x) # (B, E) -> (B, C)
        return x