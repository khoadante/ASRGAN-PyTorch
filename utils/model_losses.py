import torch
import config
from torch import nn
from typing import Union
from networks.losses import ContentLoss


def define_asrnet_loss() -> nn.L1Loss:
    pixel_criterion = nn.L1Loss()
    pixel_criterion = pixel_criterion.to(device=config.device, non_blocking=True)

    return pixel_criterion


def define_asrgan_loss() -> Union[nn.L1Loss, ContentLoss, nn.BCEWithLogitsLoss]:
    pixel_criterion = nn.L1Loss()
    content_criterion = ContentLoss(
        config.feature_model_extractor_nodes,
        config.feature_model_normalize_mean,
        config.feature_model_normalize_std,
    )
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(
        device=config.device, memory_format=torch.channels_last
    )
    content_criterion = content_criterion.to(
        device=config.device, memory_format=torch.channels_last
    )
    adversarial_criterion = adversarial_criterion.to(
        device=config.device, memory_format=torch.channels_last
    )

    return pixel_criterion, content_criterion, adversarial_criterion
