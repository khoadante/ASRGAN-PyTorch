import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.down_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False)
        )
        self.conv3 = spectral_norm(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False)
        )
        self.conv4 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)

        # Down-sampling
        down1 = self.down_block1(out1)
        down2 = self.down_block2(down1)
        down3 = self.down_block3(down2)

        # Up-sampling
        down3 = F.interpolate(
            down3, scale_factor=2, mode="bilinear", align_corners=False
        )
        up1 = self.up_block1(down3)

        up1 = torch.add(up1, down2)
        up1 = F.interpolate(up1, scale_factor=2, mode="bilinear", align_corners=False)
        up2 = self.up_block2(up1)

        up2 = torch.add(up2, down1)
        up2 = F.interpolate(up2, scale_factor=2, mode="bilinear", align_corners=False)
        up3 = self.up_block3(up2)

        up3 = torch.add(up3, out1)

        out = self.conv2(up3)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
