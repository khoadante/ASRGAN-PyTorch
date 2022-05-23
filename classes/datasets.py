import os
import cv2
import math
import torch
import random
import numpy as np
from typing import List
from torch.utils.data import Dataset
import utils.image_processing as imgproc


class TrainDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address
        image_size (int): High resolution image size
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training dataset is for data enhancement,
            and the verification data set is not for data enhancement.
        degradation_model_parameters (dict): Parameter dictionary with degenerate model

    """

    def __init__(
        self, image_dir: str, upscale_factor: int, degradation_model_parameters: dict
    ) -> None:
        super(TrainDataset, self).__init__()
        # Get all image file names in folder
        self.image_file_names = [
            os.path.join(image_dir, image_file_name)
            for image_file_name in os.listdir(image_dir)
        ]
        # Define degradation model parameters
        self.parameters = degradation_model_parameters
        # Define the size of the sinc filter kernel
        self.sinc_tensor = torch.zeros(
            [self.parameters["sinc_kernel_size"], self.parameters["sinc_kernel_size"]]
        ).float()
        self.sinc_tensor[
            self.parameters["sinc_kernel_size"] // 2,
            self.parameters["sinc_kernel_size"] // 2,
        ] = 1
        # How many times the high-resolution image is the low-resolution image
        self.upscale_factor = upscale_factor

    def __getitem__(
        self, batch_index: int
    ) -> List[torch.Tensor]:
        # Read a batch of image data
        image = (
            cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(
                np.float32
            )
            / 255.0
        )

        # Image data augmentation
        hr_image = imgproc.random_rotate(image, angles=[0, 90, 180, 270])
        hr_image = imgproc.random_horizontally_flip(hr_image, p=0.5)
        hr_image = imgproc.random_vertically_flip(hr_image, p=0.5)

        # BGR convert to RGB
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        # First degenerate operation
        kernel_size1 = random.choice(self.parameters["gaussian_kernel_range"])
        if np.random.uniform() < self.parameters["sinc_kernel_probability1"]:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size1 < int(np.median(self.parameters["gaussian_kernel_range"])):
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel1 = imgproc.sinc_kernel(omega_c, kernel_size1, pad_to=False)
        else:
            kernel1 = imgproc.random_mixed_kernels(
                self.parameters["gaussian_kernel_type"],
                self.parameters["gaussian_kernel_probability1"],
                kernel_size1,
                self.parameters["gaussian_sigma_range1"],
                self.parameters["gaussian_sigma_range1"],
                [-math.pi, math.pi],
                self.parameters["generalized_kernel_beta_range1"],
                self.parameters["plateau_kernel_beta_range1"],
                noise_range=None,
            )
        # pad kernel
        pad_size = (self.parameters["gaussian_kernel_range"][-1] - kernel_size1) // 2
        kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))

        # Second degenerate operation
        kernel_size2 = random.choice(self.parameters["gaussian_kernel_range"])
        if np.random.uniform() < self.parameters["sinc_kernel_probability2"]:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size2 < int(np.median(self.parameters["gaussian_kernel_range"])):
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = imgproc.sinc_kernel(omega_c, kernel_size2, pad_to=False)
        else:
            kernel2 = imgproc.random_mixed_kernels(
                self.parameters["gaussian_kernel_type"],
                self.parameters["gaussian_kernel_probability2"],
                kernel_size2,
                self.parameters["gaussian_sigma_range2"],
                self.parameters["gaussian_sigma_range2"],
                [-math.pi, math.pi],
                self.parameters["generalized_kernel_beta_range2"],
                self.parameters["plateau_kernel_beta_range2"],
                noise_range=None,
            )

        # pad kernel
        pad_size = (self.parameters["gaussian_kernel_range"][-1] - kernel_size2) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # Final sinc kernel
        if np.random.uniform() < self.parameters["sinc_kernel_probability3"]:
            kernel_size2 = random.choice(self.parameters["gaussian_kernel_range"])
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = imgproc.sinc_kernel(
                omega_c, kernel_size2, pad_to=self.parameters["sinc_kernel_size"]
            )
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.sinc_tensor

        kernel1 = torch.FloatTensor(kernel1)
        kernel2 = torch.FloatTensor(kernel2)
        sinc_kernel = torch.FloatTensor(sinc_kernel)

        return {
            "hr": hr_tensor,
            "kernel1": kernel1,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
        }

    def __len__(self) -> int:
        return len(self.image_file_names)


class ValidDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address
        image_size (int): High resolution image size
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training dataset is for data enhancement,
            and the verification data set is not for data enhancement.
        degradation_model_parameters (dict): Parameter dictionary with degenerate model

    """

    def __init__(
        self,
        image_dir: str,
        image_size: int,
        upscale_factor: int,
        degradation_model_parameters: dict,
    ) -> None:
        super(ValidDataset, self).__init__()
        # Get all image file names in folder
        self.image_file_names = [
            os.path.join(image_dir, image_file_name)
            for image_file_name in os.listdir(image_dir)
        ]
        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size
        # Define degradation model parameters
        self.parameters = degradation_model_parameters
        # Define the size of the sinc filter kernel
        self.sinc_tensor = torch.zeros(
            [self.parameters["sinc_kernel_size"], self.parameters["sinc_kernel_size"]]
        ).float()
        self.sinc_tensor[
            self.parameters["sinc_kernel_size"] // 2,
            self.parameters["sinc_kernel_size"] // 2,
        ] = 1
        # How many times the high-resolution image is the low-resolution image
        self.upscale_factor = upscale_factor

    def __getitem__(
        self, batch_index: int
    ) -> List[torch.Tensor]:
        # Read a batch of image data
        image = (
            cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(
                np.float32
            )
            / 255.0
        )

        # Center crop image
        hr_image = imgproc.center_crop(image, self.image_size)
        # Use Bicubic kernel create LR image
        lr_image = imgproc.image_resize(hr_image, 1 / self.upscale_factor)

        # BGR convert to RGB
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TestDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_lr_image_dir (str): Test dataset address for low resolution image dir.
        test_hr_image_dir (str): Test dataset address for high resolution image dir.
    """

    def __init__(self, test_lr_image_dir: str, test_hr_image_dir: str) -> None:
        super(TestDataset, self).__init__()
        # Get all image file names in folder
        self.lr_image_file_names = [
            os.path.join(test_lr_image_dir, x) for x in os.listdir(test_lr_image_dir)
        ]
        self.hr_image_file_names = [
            os.path.join(test_hr_image_dir, x) for x in os.listdir(test_lr_image_dir)
        ]

    def __getitem__(self, batch_index: int) -> List[torch.Tensor]:
        # Read a batch of image data
        lr_image = (
            cv2.imread(
                self.lr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED
            ).astype(np.float32)
            / 255.0
        )
        hr_image = (
            cv2.imread(
                self.hr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED
            ).astype(np.float32)
            / 255.0
        )

        # BGR convert to RGB
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.lr_image_file_names)
