import os

num_worker = os.cpu_count()

os.system("pip install -r requirements.txt")
os.system("python scripts/download_dataset.py")
os.system(
    f"python scripts/prepare_dataset.py --images_dir datasets/DIV2K/DIV2K_train_HR --output_dir datasets/DIV2K/ASRGAN/train --image_size 510 --step 255 --num_workers {num_worker}"
)
os.system(
    f"python scripts/prepare_dataset.py --images_dir datasets/DIV2K/DIV2K_valid_HR --output_dir datasets/DIV2K/ASRGAN/valid --image_size 510 --step 510 --num_workers {num_worker}"
)
