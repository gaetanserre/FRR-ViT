import torch as th
import torch.nn as nn
from collections import OrderedDict

class EncoderBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, mlp_size):
    super(EncoderBlock, self).__init__()

    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    self.mlp = nn.Sequential(
        nn.Linear(embed_dim, mlp_size),
        nn.GELU(),
        nn.Linear(mlp_size, embed_dim)
    )
  
  def forward(self, input):
    x = self.norm1(input)
    x = self.attn(x, x, x)[0] + input
    x = self.mlp(self.norm2(x)) + x
    return x


class VisionTransformer(nn.Module):
  def __init__(self, img_size, num_classes, in_chans, patch_size, embed_dim, heads, mlp_size, layers):
    super(VisionTransformer, self).__init__()

    self.patch_embedding = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    self.cls_token = nn.Parameter(th.zeros(1, 1, embed_dim))

    seq_len = (img_size // patch_size) ** 2 + 1 #+1 for the cls_token
    self.pos_embedding = nn.Parameter(th.empty(1, seq_len, embed_dim).normal_(std=0.02))

    blocks = OrderedDict()
    for i in range(layers):
      blocks[f"encoder_block{i}"] = EncoderBlock(embed_dim, heads, mlp_size)

    self.encoder = nn.Sequential(blocks)

    self.ln = nn.LayerNorm(embed_dim)

    self.head = nn.Linear(embed_dim, num_classes)
    
  def forward(self, x):
    # Patch the images
    x = self.patch_embedding(x)
    x = th.flatten(x, start_dim=2)
    x = th.einsum("b c h -> b h c", x)

    # Add the cls_token
    x = th.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

    # Add the positional embedding
    x = x + self.pos_embedding

    # Transformer
    x = self.ln(self.encoder(x))

    # Classification
    x = x[:, 0]
    x = self.head(x)
    return x

def vit_b_16(num_classes):
  return VisionTransformer(
    img_size=224,
    num_classes=num_classes,
    in_chans=3,
    patch_size=16,
    embed_dim=768,
    heads=12,
    mlp_size=3072,
    layers=12
  )

def vit_l_16(num_classes):
  return VisionTransformer(
    img_size=224,
    num_classes=num_classes,
    in_chans=3,
    patch_size=16,
    embed_dim=1024,
    heads=16,
    mlp_size=4096,
    layers=24
  )

def vit_h_14(num_classes):
  return VisionTransformer(
    img_size=224,
    num_classes=num_classes,
    in_chans=3,
    patch_size=14,
    embed_dim=1280,
    heads=16,
    mlp_size=5120,
    layers=32
  )