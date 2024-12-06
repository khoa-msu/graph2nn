from pycls.config import cfg

import pycls.utils.logging as lu
from .relation_graph import *

logger = lu.get_logger(__name__)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4.0, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels=3, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for cls token
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # Patch embedding
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # Add class token
        x += self.pos_embed  # Add positional embedding
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])  # Take the class token
        return self.head(x)

def build_vit():
    return VisionTransformer(
        img_size=cfg.VIT.IMG_SIZE,
        patch_size=cfg.VIT.PATCH_SIZE,
        num_classes=cfg.MODEL.NUM_CLASSES,
        embed_dim=cfg.VIT.EMBED_DIM,
        depth=cfg.VIT.DEPTH,
        num_heads=cfg.VIT.NUM_HEADS,
        mlp_ratio=cfg.VIT.MLP_RATIO,
        dropout=cfg.VIT.DROPOUT,
    )
