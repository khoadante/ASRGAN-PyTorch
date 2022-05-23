import config
from torch import optim
from typing import List


def define_asrnet_scheduler(optimizer) -> optim.lr_scheduler.StepLR:
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma
    )

    return scheduler


def define_asrgan_scheduler(
    g_optimizer: optim.AdamW, d_optimizer: optim.AdamW
) -> List[optim.lr_scheduler.MultiStepLR]:
    g_scheduler = optim.lr_scheduler.MultiStepLR(
        g_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma
    )
    d_scheduler = optim.lr_scheduler.MultiStepLR(
        d_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma
    )

    return g_scheduler, d_scheduler
