import config
from typing import List
from torch.utils.data import DataLoader
from classes.prefetchers import CUDAPrefetcher
from classes.datasets import TrainDataset, ValidDataset, TestDataset


def load_datasets() -> List[CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainDataset(
        config.train_image_dir,
        config.upscale_factor,
        config.degradation_model_parameters,
    )
    valid_datasets = ValidDataset(
        config.valid_image_dir,
        config.image_size,
        config.upscale_factor,
        config.degradation_model_parameters,
    )
    test_datasets = TestDataset(config.test_lr_image_dir, config.test_hr_image_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(
        train_datasets,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    valid_dataloader = DataLoader(
        valid_datasets,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, valid_prefetcher, test_prefetcher
