import torch
from torch import nn
import torch.nn.functional as F
from networks.blocks import ResidualDenseBlock, ResidualResidualDenseBlock

class Generator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int) -> None:
        super(Generator, self).__init__()
        if upscale_factor == 2:
            in_channels *= 4
            downscale_factor = 2
        elif upscale_factor == 1:
            in_channels *= 16
            downscale_factor = 4
        else:
            in_channels *= 1
            downscale_factor = 1

        # Down-sampling layer
        self.downsampling = nn.PixelUnshuffle(downscale_factor)

        # The first layer of convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network
        trunk = []
        for _ in range(23):
            trunk.append(ResidualResidualDenseBlock(64, 32))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer
        self.upsampling1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        self.upsampling2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Output layer
        self.conv4 = nn.Conv2d(64, out_channels, (3, 3), (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # If upscale_factor not equal 4, must use nn.PixelUnshuffle() ops
        out = self.downsampling(x)

        out1 = self.conv1(out)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
