import os

os.system("pip install -r requirements.txt")
os.system("python scripts/download_dataset.py")
os.system(
    "python scripts/prepare_dataset.py --images_dir datasets/DIV2K/DIV2K_train_HR --output_dir datasets/DIV2K/ASRGAN/train --image_size 400 --step 200 --num_workers 16"
)
os.system(
    "python scripts/prepare_dataset.py --images_dir datasets/DIV2K/DIV2K_valid_HR --output_dir datasets/DIV2K/ASRGAN/valid --image_size 400 --step 400 --num_workers 16"
)
