from typing import Tuple

import torch
import torch.nn as nn


class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, output_padding: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channels: int, height: int, width: int) -> None:
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), self.channels, self.height, self.width)


def compute_conv_output_size(input_size: Tuple[int, int], num_layers: int, kernel_size: int = 4, stride: int = 2, padding: int = 1) -> Tuple[int, int]:
    height, width = input_size
    for _ in range(num_layers):
        height = (height + 2 * padding - kernel_size) // stride + 1
        width = (width + 2 * padding - kernel_size) // stride + 1
    return height, width
