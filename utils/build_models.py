import torch
import config
from torch import nn
from typing import List
from networks.models import Generator, Discriminator


def build_asrnet_model() -> nn.Module:
    model = Generator(config.in_channels, config.out_channels, config.upscale_factor)
    model = model.to(device=config.device, memory_format=torch.channels_last)

    return model


def build_asrgan_model() -> List[nn.Module, nn.Module]:
    generator = Generator(
        config.in_channels, config.out_channels, config.upscale_factor
    )
    discriminator = Discriminator()
    # Transfer to CUDA
    generator = generator.to(device=config.device, memory_format=torch.channels_last)
    discriminator = discriminator.to(
        device=config.device, memory_format=torch.channels_last
    )

    return generator, discriminator
