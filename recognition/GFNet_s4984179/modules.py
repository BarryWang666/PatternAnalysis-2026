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
