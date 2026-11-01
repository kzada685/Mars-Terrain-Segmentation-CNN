"""Initial CNN for four-class segmentation of grayscale Mars rover images."""

from torch import nn
from torch.nn import functional as F


def convolution_block(input_channels, output_channels):
    """Learn spatial features without changing the height or width."""
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


class TerrainCNN(nn.Module):
    """A basic encoder-decoder; skip-connection comparisons come in a later step.

    Input: grayscale images shaped (batch, 1, height, width).
    Output: raw class scores shaped (batch, 4, height, width), ordered as
    soil, bedrock, sand, and large rock. No softmax is applied here.
    """

    def __init__(self):
        super().__init__()

        # Encoder: increase feature channels while reducing spatial resolution.
        self.encoder1 = convolution_block(1, 16)
        self.encoder2 = convolution_block(16, 32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = convolution_block(32, 64)

        # Decoder: restore image resolution, then output four scores per pixel.
        self.decoder2 = convolution_block(64, 32)
        self.decoder1 = convolution_block(32, 16)
        self.classifier = nn.Conv2d(16, 4, kernel_size=1)

    def forward(self, image):
        first_features = self.encoder1(image)
        second_features = self.encoder2(self.pool(first_features))
        encoded = self.bottleneck(self.pool(second_features))

        decoded = F.interpolate(
            encoded,
            size=second_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.decoder2(decoded)

        decoded = F.interpolate(
            decoded,
            size=first_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.decoder1(decoded)
        return self.classifier(decoded)
