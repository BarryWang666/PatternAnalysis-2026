"""COMP3710 | s4984179 | Scratch CNN and rectangular GFNet components.

Input: float32 grayscale slices [B,1,240,256]. Classifiers output NC/AD logits.
GFNet source: https://github.com/raoyongming/GFNet
Stochastic depth source: https://github.com/huggingface/pytorch-image-models
"""


import torch


from torch import nn


class CNNBlock(nn.Module):
    """Two convolution/GroupNorm/GELU layers, then halve spatial dimensions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

    def forward(self, images):
        return self.layers(images)


class BaselineCNN(nn.Module):
    """Four stages map native [B,1,240,256] slices to NC/AD logits [B,2].

    Input pixels are already scaled by the loader. Global pooling avoids a
    large flattened-image head; cross-entropy consumes the output directly.
    """

    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(
            CNNBlock(1, 32), CNNBlock(32, 64),
            CNNBlock(64, 128), CNNBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(256, 2),
        )

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 240, 256):
            raise ValueError("Expected slices shaped [B,1,240,256]")
        return self.head(self.stages(images))


class PatchEmbed(nn.Module):
    """Project [B,1,240,256] slices to row-major [B,240,256] patch tokens.

    Kernel/stride 16 preserve a rectangular 15x16 grid without padding.
    """

    def __init__(self):
        super().__init__()
        self.grid_size = (15, 16)
        self.num_patches = 240
        self.proj = nn.Conv2d(1, 256, kernel_size=16, stride=16)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 240, 256):
            raise ValueError("Expected slices shaped [B,1,240,256]")
        return self.proj(images).flatten(2).transpose(1, 2)


class GlobalFilter(nn.Module):
    """Learn a complex frequency filter on an explicit rectangular token grid.

    Adapted from Rao et al., Global Filter Networks (2021):
    https://github.com/raoyongming/GFNet/blob/master/gfnet.py
    Input/output: real [B,H*W,C] tokens; weights store H*(W//2+1)*C complex
    coefficients as pairs of real scalars. FFT axes are spatial, not channels.
    """

    def __init__(self, grid_size, dim):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.dim = dim
        height, width = self.grid_size
        self.complex_weight = nn.Parameter(
            torch.randn(height, width // 2 + 1, dim, 2) * 0.02,
        )

    def forward(self, tokens):
        height, width = self.grid_size
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (height * width, self.dim):
            raise ValueError("Tokens must match the declared grid and feature width")
        grid = tokens.reshape(tokens.shape[0], height, width, self.dim)
        spectrum = torch.fft.rfft2(grid, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight.contiguous())
        grid = torch.fft.irfft2(
            spectrum * weight, s=(height, width), dim=(1, 2), norm="ortho",
        )
        return grid.reshape(tokens.shape[0], height * width, self.dim)


class MLP(nn.Module):
    """Expand and contract each token's feature width with GELU."""

    def __init__(self, dim=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, tokens):
        return self.fc2(self.act(self.fc1(tokens)))


class DropPath(nn.Module):
    """Per-example stochastic depth, following timm's drop_path implementation."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError("DropPath probability must be in [0,1)")
        self.drop_prob = drop_prob

    def forward(self, tokens):
        if not self.training or self.drop_prob == 0:
            return tokens
        keep = 1 - self.drop_prob
        shape = (tokens.shape[0],) + (1,) * (tokens.ndim - 1)
        gate = (keep + torch.rand(shape, dtype=tokens.dtype, device=tokens.device)).floor()
        return tokens * gate / keep


class GFNetBlock(nn.Module):
    """GFNet's norm/filter/norm/MLP branch with one residual connection."""

    def __init__(self, grid_size=(15, 16), dim=256, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.filter = GlobalFilter(grid_size, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens):
        branch = self.mlp(self.norm2(self.filter(self.norm1(tokens))))
        return tokens + self.drop_path(branch)


class GFNet(nn.Module):
    """Classify [B,1,240,256] slices with twelve width-256 GFNet blocks.

    Position embeddings preserve patch location. Token averaging and a linear
    head return raw NC/AD logits [B,2]; evaluation disables stochastic depth.
    """

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, 240, 256))
        self.blocks = nn.ModuleList(
            GFNetBlock(self.patch_embed.grid_size, 256, i * 0.1 / 11)
            for i in range(12)
        )
        self.norm = nn.LayerNorm(256, eps=1e-6)
        self.head = nn.Linear(256, 2)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        tokens = self.patch_embed(images) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens).mean(dim=1))
