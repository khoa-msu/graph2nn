import torch
import torch.nn as nn
import math


# Helper function to create patches
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Convert image into patches and flatten them
        x = self.projection(x)  # Shape: [batch_size, emb_size, num_patches ** 0.5, num_patches ** 0.5]
        x = x.flatten(2)  # Shape: [batch_size, emb_size, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, emb_size]
        return x


# Positional encoding to preserve information about the position of the patches
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, num_patches, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(num_patches, emb_size)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Multi-head self-attention for transformer encoder
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        assert self.head_dim * num_heads == emb_size, "Embedding size must be divisible by num_heads"

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.nn.functional.softmax(energy, dim=-1)

        out = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, -1,
                                                                           self.num_heads * self.head_dim)
        return self.fc_out(out)


# Feedforward network for the Transformer encoder
class FeedForward(nn.Module):
    def __init__(self, emb_size, expansion=4, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_size, expansion * emb_size)
        self.fc2 = nn.Linear(expansion * emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.1, expansion=4):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(emb_size, num_heads)
        self.ffn = FeedForward(emb_size, expansion, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Attention
        attn_output = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# Vision Transformer (ViT)
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, emb_size=768, num_heads=12,
                 num_layers=12, dropout=0.1, expansion=4):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_encoding = PositionalEncoding(emb_size, (img_size // patch_size) ** 2 + 1, dropout)

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, dropout, expansion) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)

        # Class token prepended to sequence
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, emb_size]
        x = torch.cat((cls_tokens, x), dim=1)  # Add class token

        # Add positional encodings
        x = self.pos_encoding(x)

        # Transformer encoder blocks
        for encoder in self.transformer_encoders:
            x = encoder(x)

        # Classification head (using the class token's output)
        cls_output = x[:, 0]  # Take the output of the class token
        return self.fc_out(cls_output)


# Example usage:
img = torch.randn(8, 3, 224, 224)  # Batch of 8 images, 3 channels, 224x224 pixels
model = VisionTransformer()
output = model(img)
print(output.shape)  # Expected output: [8, 1000] (8 samples, 1000 classes)
