import config
from torch import nn
from torch import optim
from typing import List


def define_asrnet_optimizer(model) -> optim.AdamW:
    optimizer = optim.AdamW(model.parameters(), config.model_lr, config.model_betas)

    return optimizer


def define_asrgan_optimizer(
    generator: nn.Module, discriminator: nn.Module
) -> List[optim.AdamW]:
    g_optimizer = optim.AdamW(
        generator.parameters(), config.model_lr, config.model_betas
    )
    d_optimizer = optim.AdamW(
        discriminator.parameters(), config.model_lr, config.model_betas
    )

    return g_optimizer, d_optimizer
