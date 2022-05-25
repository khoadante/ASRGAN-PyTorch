import os
import time
import torch
import config
import random
import shutil
import numpy as np
from typing import List
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.nn import functional as F
from networks.models import EMA
from networks.losses import ContentLoss, GANLoss
from classes.prefetchers import CUDAPrefetcher
from utils.image_metrics import NIQE
from utils.load_datasets import load_datasets
from utils.build_models import build_asrgan_model
from utils.model_losses import define_asrgan_loss
from utils.model_optimizers import define_asrgan_optimizer
from utils.model_schedulers import define_asrgan_scheduler
from utils.training_meters import AverageMeter, ProgressMeter
from torch.utils.tensorboard import SummaryWriter
import utils.image_processing as imgproc


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_niqe = 100.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_datasets()
    print("Load dataset successfully.")

    generator, discriminator = build_asrgan_model()
    print("Build ASRGAN model successfully.")

    pixel_criterion, content_criterion, adversarial_criterion = define_asrgan_loss()
    print("Define all loss functions successfully.")

    g_optimizer, d_optimizer = define_asrgan_optimizer(generator, discriminator)
    print("Define all optimizer functions successfully.")

    g_scheduler, d_scheduler = define_asrgan_scheduler(g_optimizer, d_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    if config.resume:
        print("Loading ASRNet model weights")
        # Load checkpoint model
        checkpoint = torch.load(
            config.resume, map_location=lambda storage, loc: storage
        )
        generator.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Loaded ASRNet model weights.")

    print("Check whether the pretrained discriminator model is restored...")
    if config.resume_d:
        # Load checkpoint model
        checkpoint = torch.load(
            config.resume_d, map_location=lambda storage, loc: storage
        )
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_niqe = checkpoint["best_niqe"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = discriminator.state_dict()
        new_state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k in model_state_dict.keys()
        }
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        discriminator.load_state_dict(model_state_dict)
        # Load the optimizer model
        d_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        d_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained discriminator model weights.")

    print("Check whether the pretrained generator model is restored...")
    if config.resume_g:
        # Load checkpoint model
        checkpoint = torch.load(
            config.resume_g, map_location=lambda storage, loc: storage
        )
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_niqe = checkpoint["best_niqe"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = generator.state_dict()
        new_state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k in model_state_dict.keys()
        }
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        generator.load_state_dict(model_state_dict)
        # Load the optimizer model
        g_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        g_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained generator model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler.
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    niqe_model = NIQE(config.upscale_factor, config.niqe_model_path)

    # Transfer the IQA model to the specified device
    niqe_model = niqe_model.to(device=config.device, non_blocking=True)

    # Create an Exponential Moving Average Model
    ema_model = EMA(generator, config.ema_model_weight_decay)
    ema_model = ema_model.to(device=config.device, non_blocking=True)
    ema_model.register()

    for epoch in range(start_epoch, config.epochs):
        train(
            discriminator,
            generator,
            ema_model,
            train_prefetcher,
            pixel_criterion,
            content_criterion,
            adversarial_criterion,
            d_optimizer,
            g_optimizer,
            epoch,
            scaler,
            writer,
        )
        _ = validate(
            generator, ema_model, valid_prefetcher, epoch, writer, niqe_model, "Valid"
        )
        niqe = validate(
            generator, ema_model, test_prefetcher, epoch, writer, niqe_model, "Test"
        )
        print("\n")

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

        # Automatically save the model with the highest index
        is_best = niqe < best_niqe
        best_niqe = min(niqe, best_niqe)
        torch.save(
            {
                "epoch": epoch + 1,
                "best_niqe": best_niqe,
                "state_dict": discriminator.state_dict(),
                "optimizer": d_optimizer.state_dict(),
                "scheduler": d_scheduler.state_dict(),
            },
            os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"),
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "best_niqe": best_niqe,
                "state_dict": ema_model.state_dict(),
                "optimizer": g_optimizer.state_dict(),
                "scheduler": g_scheduler.state_dict(),
            },
            os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
        )
        if is_best:
            shutil.copyfile(
                os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "d_best.pth.tar"),
            )
            shutil.copyfile(
                os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "g_best.pth.tar"),
            )
        if (epoch + 1) == config.epochs:
            shutil.copyfile(
                os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "d_last.pth.tar"),
            )
            shutil.copyfile(
                os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "g_last.pth.tar"),
            )


def train(
    discriminator: nn.Module,
    generator: nn.Module,
    ema_model: nn.Module,
    train_prefetcher: CUDAPrefetcher,
    pixel_criterion: nn.L1Loss,
    content_criterion: ContentLoss,
    adversarial_criterion: GANLoss,
    d_optimizer: optim.Adam,
    g_optimizer: optim.Adam,
    epoch: int,
    scaler: amp.GradScaler,
    writer: SummaryWriter,
) -> None:
    """Training main program

    Args:
        discriminator (nn.Module): discriminator model in adversarial networks
        generator (nn.Module): generator model in adversarial networks
        ema_model (nn.Module): Exponential Moving Average Model
        train_prefetcher (CUDAPrefetcher): training dataset iterator
        pixel_criterion (nn.L1Loss): Calculate the pixel difference between real and fake samples
        content_criterion (ContentLoss): Calculate the feature difference between real samples and fake samples by the feature extraction model
        adversarial_criterion (nn.BCEWithLogitsLoss): Calculate the semantic difference between real samples and fake samples by the discriminator model
        d_optimizer (optim.Adam): an optimizer for optimizing discriminator models in adversarial networks
        g_optimizer (optim.Adam): an optimizer for optimizing generator models in adversarial networks
        epoch (int): number of training epochs during training the adversarial network
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Defining JPEG image manipulation methods
    jpeg_operation = imgproc.DiffJPEG(differentiable=False)
    jpeg_operation = jpeg_operation.to(device=config.device, non_blocking=True)
    # Define image sharpening method
    usm_sharpener = imgproc.USMSharp()
    usm_sharpener = usm_sharpener.to(device=config.device, non_blocking=True)

    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)

    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    progress = ProgressMeter(
        batches,
        [
            batch_time,
            data_time,
            pixel_losses,
            content_losses,
            adversarial_losses,
            d_hr_probabilities,
            d_sr_probabilities,
        ],
        prefix=f"Epoch: [{epoch + 1}]",
    )

    # Put all model in train mode.
    discriminator.train()
    generator.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        hr = batch_data["hr"].to(device=config.device, non_blocking=True)
        kernel1 = batch_data["kernel1"].to(device=config.device, non_blocking=True)
        kernel2 = batch_data["kernel2"].to(device=config.device, non_blocking=True)
        sinc_kernel = batch_data["sinc_kernel"].to(
            device=config.device, non_blocking=True
        )

        # Sharpen high-resolution images
        out = usm_sharpener(hr)

        # Get original image size
        image_height, image_width = out.size()[2:4]

        # First degradation process
        # Gaussian blur
        if (
            np.random.uniform()
            <= config.degradation_process_parameters["first_blur_probability"]
        ):
            out = imgproc.blur(out, kernel1)

        # Resize
        updown_type = random.choices(
            ["up", "down", "keep"],
            config.degradation_process_parameters["resize_probability1"],
        )[0]
        if updown_type == "up":
            scale = np.random.uniform(
                1, config.degradation_process_parameters["resize_range1"][1]
            )
        elif updown_type == "down":
            scale = np.random.uniform(
                config.degradation_process_parameters["resize_range1"][0], 1
            )
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # Noise
        if (
            np.random.uniform()
            < config.degradation_process_parameters["gaussian_noise_probability1"]
        ):
            out = imgproc.random_add_gaussian_noise_pt(
                image=out,
                sigma_range=config.degradation_process_parameters["noise_range1"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters[
                    "gray_noise_probability1"
                ],
            )
        else:
            out = imgproc.random_add_poisson_noise_pt(
                image=out,
                scale_range=config.degradation_process_parameters[
                    "poisson_scale_range1"
                ],
                gray_prob=config.degradation_process_parameters[
                    "gray_noise_probability1"
                ],
                clip=True,
                rounds=False,
            )

        # JPEG
        quality = out.new_zeros(out.size(0)).uniform_(
            *config.degradation_process_parameters["jpeg_range1"]
        )
        out = torch.clamp(
            out, 0, 1
        )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeg_operation(out, quality=quality)

        # Second degradation process
        # Gaussian blur
        if (
            np.random.uniform()
            < config.degradation_process_parameters["second_blur_probability"]
        ):
            out = imgproc.blur(out, kernel2)

        # Resize
        updown_type = random.choices(
            ["up", "down", "keep"],
            config.degradation_process_parameters["resize_probability2"],
        )[0]
        if updown_type == "up":
            scale = np.random.uniform(
                1, config.degradation_process_parameters["resize_range2"][1]
            )
        elif updown_type == "down":
            scale = np.random.uniform(
                config.degradation_process_parameters["resize_range2"][0], 1
            )
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out,
            size=(
                int(image_height / config.upscale_factor * scale),
                int(image_width / config.upscale_factor * scale),
            ),
            mode=mode,
        )

        # Noise
        if (
            np.random.uniform()
            < config.degradation_process_parameters["gaussian_noise_probability2"]
        ):
            out = imgproc.random_add_gaussian_noise_pt(
                image=out,
                sigma_range=config.degradation_process_parameters["noise_range2"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters[
                    "gray_noise_probability2"
                ],
            )
        else:
            out = imgproc.random_add_poisson_noise_pt(
                image=out,
                scale_range=config.degradation_process_parameters[
                    "poisson_scale_range2"
                ],
                gray_prob=config.degradation_process_parameters[
                    "gray_noise_probability2"
                ],
                clip=True,
                rounds=False,
            )

        if np.random.uniform() < 0.5:
            # Resize
            out = F.interpolate(
                out,
                size=(
                    image_height // config.upscale_factor,
                    image_width // config.upscale_factor,
                ),
                mode=random.choice(["area", "bilinear", "bicubic"]),
            )
            # Sinc blur
            out = imgproc.blur(out, sinc_kernel)

            # JPEG
            quality = out.new_zeros(out.size(0)).uniform_(
                *config.degradation_process_parameters["jpeg_range2"]
            )
            out = torch.clamp(out, 0, 1)
            out = jpeg_operation(out, quality=quality)
        else:
            # JPEG
            quality = out.new_zeros(out.size(0)).uniform_(
                *config.degradation_process_parameters["jpeg_range2"]
            )
            out = torch.clamp(out, 0, 1)
            out = jpeg_operation(out, quality=quality)

            # Resize
            out = F.interpolate(
                out,
                size=(
                    image_height // config.upscale_factor,
                    image_width // config.upscale_factor,
                ),
                mode=random.choice(["area", "bilinear", "bicubic"]),
            )

            # Sinc blur
            out = imgproc.blur(out, sinc_kernel)

        # Clamp and round
        lr = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        # LR and HR crop the specified area respectively
        lr, hr = imgproc.random_crop(lr, hr, config.image_size, config.upscale_factor)

        # Set the real sample label to 1, and the false sample label to 0
        real_label = torch.full(
            [lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device
        )
        fake_label = torch.full(
            [lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device
        )

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        generator.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        with amp.autocast():
            sr = generator(lr)
            pixel_loss = config.pixel_weight * pixel_criterion(usm_sharpener(sr), hr)
            content_loss = torch.sum(
                torch.multiply(
                    torch.tensor(config.content_weight),
                    torch.tensor(content_criterion(usm_sharpener(sr), hr)),
                )
            )
            fake_output = discriminator(sr.detach())
            adversarial_loss = 0.1 * adversarial_criterion(fake_output, True)
            # adversarial_loss = config.adversarial_weight * adversarial_criterion(discriminator(sr), True)
            # Calculate the generator total loss value
            g_loss = pixel_loss + content_loss + adversarial_loss
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        scaler.scale(g_loss).backward()
        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        scaler.step(g_optimizer)
        scaler.update()
        # Finish training the generator model

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        discriminator.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        with amp.autocast():
            hr_output = discriminator(hr)
            d_loss_hr = adversarial_criterion(hr_output, real_label)
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the real sample
        scaler.scale(d_loss_hr).backward()

        # Calculate the classification score of the discriminator model for fake samples
        with amp.autocast():
            # Use the generator model to generate fake samples
            sr = generator(lr)
            sr_output = discriminator(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
            # Calculate the total discriminator loss value
            d_loss = d_loss_sr + d_loss_hr
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        scaler.scale(d_loss_sr).backward()
        # Improve the discriminator model's ability to classify real and fake samples
        scaler.step(d_optimizer)
        scaler.update()
        # Finish training the discriminator model

        # Update EMA
        ema_model.update()

        # Calculate the score of the discriminator on real samples and fake samples, the score of real samples is close to 1, and the score of fake samples is close to 0
        d_hr_probability = torch.sigmoid_(torch.mean(hr_output.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))

        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
    model: nn.Module,
    ema_model: nn.Module,
    data_prefetcher: CUDAPrefetcher,
    epoch: int,
    writer: SummaryWriter,
    niqe_model: nn.Module,
    mode: str,
) -> float:
    """Test main program

    Args:
        model (nn.Module): generator model in adversarial networks
        ema_model (nn.Module): Exponential Moving Average Model
        data_prefetcher (CUDAPrefetcher): test dataset iterator
        epoch (int): number of test epochs during training of the adversarial network
        writer (SummaryWriter): log file management function
        niqe_model (nn.Module): The model used to calculate the model NIQE metric
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    niqe_metrics = AverageMeter("NIQE", ":4.2f")
    progress = ProgressMeter(
        len(data_prefetcher), [batch_time, niqe_metrics], prefix=f"{mode}: "
    )

    # Restore the model before the EMA
    ema_model.apply_shadow()
    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)
            hr = batch_data["hr"].to(device=config.device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # Statistical loss value for terminal data output
            niqe = niqe_model(sr)
            niqe_metrics.update(niqe.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % (batches // 5) == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches
            # to ensure that the terminal prints data normally
            batch_index += 1

    # Restoring the EMA model
    ema_model.restore()

    # Print average PSNR metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/NIQE", niqe_metrics.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return niqe_metrics.avg


if __name__ == "__main__":
    main()
