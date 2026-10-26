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


class ConvStem(nn.Module):
    """Learn overlapping local features before the 15x16 Fourier token grid.

    Four padded 3x3 stride-two convolutions preserve the native rectangular
    canvas through 120x128, 60x64, 30x32 and 15x16 feature maps.
    """

    def __init__(self):
        super().__init__()
        self.grid_size = (15, 16)
        self.num_patches = 240
        layers = []
        for incoming, outgoing in zip((1, 32, 64, 128), (32, 64, 128, 256)):
            layers.extend((
                nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, outgoing), nn.GELU(),
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 240, 256):
            raise ValueError("Expected slices shaped [B,1,240,256]")
        return self.layers(images).flatten(2).transpose(1, 2)


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
    """Expand each token by the declared ratio, then contract with GELU."""

    def __init__(self, dim=256, ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, ratio * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ratio * dim, dim)

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

    def __init__(self, grid_size=(15, 16), dim=256, drop_path=0.0, layer_scale=None,
                 mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.filter = GlobalFilter(grid_size, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)
        self.gamma = (nn.Parameter(torch.full((dim,), layer_scale))
                      if layer_scale is not None else None)

    def forward(self, tokens):
        branch = self.mlp(self.norm2(self.filter(self.norm1(tokens))))
        if self.gamma is not None:
            branch = self.gamma * branch
        return tokens + self.drop_path(branch)


class GFNet(nn.Module):
    """Classify [B,1,240,256] slices with configurable width-256 GFNet blocks.

    Position embeddings preserve patch location. Token averaging and a linear
    head return raw NC/AD logits [B,2]; evaluation disables stochastic depth.
    """

    def __init__(self, embedding="patch", depth=12, mlp_ratio=4):
        super().__init__()
        if embedding not in ("patch", "conv"):
            raise ValueError("Choose patch or conv embedding")
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("GFNet depth must be a positive integer")
        if not isinstance(mlp_ratio, int) or mlp_ratio not in (2, 4):
            raise ValueError("Choose MLP expansion ratio 2 or 4")
        self.patch_embed = PatchEmbed() if embedding == "patch" else ConvStem()
        self.pos_embed = nn.Parameter(torch.zeros(1, 240, 256))
        self.blocks = nn.ModuleList(
            GFNetBlock(self.patch_embed.grid_size, 256, i * 0.1 / max(1, depth - 1),
                       mlp_ratio=mlp_ratio)
            for i in range(depth)
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


class GFNetPyramid(nn.Module):
    """Three native rectangular stages, adapted from reference GFNetPyramid.

    Source: https://github.com/shakes76/GFNet/blob/master/gfnet.py
    Overlapping local convolutions give 60x64/64, 30x32/128 and 15x16/256
    features. Fourier processing precedes the next downsampling operation.
    Three stages retain the native canvas without a fourth odd-height merge.
    LayerScale starts at .001 as in the reference Pyramid constructor.
    """

    def __init__(self, stage_depths=(2, 2, 2)):
        super().__init__()
        if (len(stage_depths) != 3
                or any(not isinstance(depth, int) or depth < 1 for depth in stage_depths)):
            raise ValueError("Pyramid needs three positive stage depths")
        self.grid_sizes = ((60, 64), (30, 32), (15, 16))
        self.dims = (64, 128, 256)
        local = [nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.GELU(),
        ) for incoming, outgoing in zip((1, 32, 64, 128), (32, 64, 128, 256))]
        self.stem = nn.Sequential(*local[:2])
        self.downsamples = nn.ModuleList(local[2:])
        self.pos_embed = nn.Parameter(torch.zeros(1, 60 * 64, 64))
        self.stages = nn.ModuleList()
        cursor, total = 0, sum(stage_depths)
        for grid, dim, depth in zip(self.grid_sizes, self.dims, stage_depths):
            self.stages.append(nn.Sequential(*[
                GFNetBlock(grid, dim, (cursor + index) * .1 / max(1, total - 1),
                           layer_scale=.001)
                for index in range(depth)
            ]))
            cursor += depth
        self.norm = nn.LayerNorm(256, eps=1e-6)
        self.head = nn.Linear(256, 2)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(GFNet._init_weights)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 240, 256):
            raise ValueError("Expected slices shaped [B,1,240,256]")
        features = self.stem(images)
        for index, (grid, dim, blocks) in enumerate(
                zip(self.grid_sizes, self.dims, self.stages)):
            if index:
                features = self.downsamples[index - 1](features)
            tokens = features.flatten(2).transpose(1, 2)
            if index == 0:
                tokens = tokens + self.pos_embed
            tokens = blocks(tokens)
            features = tokens.transpose(1, 2).reshape(images.shape[0], dim, *grid)
        return self.head(self.norm(tokens).mean(dim=1))
