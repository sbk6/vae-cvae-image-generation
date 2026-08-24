import torch
import torch.nn as nn

from src.models.layers import ConvEncoderBlock, Flatten, compute_conv_output_size


class DigitClassifier(nn.Module):
    """Petit CNN de classification, entraîné sur les vraies images MNIST pour
    mesurer la contrôlabilité du CVAE sans le biais du proxy plus-proche-centroïde
    (même principe que le classifieur Fashion-MNIST de David)."""

    def __init__(self, channels: int, image_size, num_classes: int, hidden_channels: int = 32) -> None:
        super().__init__()
        h, w = compute_conv_output_size(tuple(image_size), num_layers=2)
        self.features = nn.Sequential(
            ConvEncoderBlock(channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            ConvEncoderBlock(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 * h * w, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
