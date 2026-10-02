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
