# main.py
import os
import logging
import datetime
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import SinogramDataset, split_train_test
from model import UNet
from train import train_model
from logger_utils import setup_logging

# 1) We define or import the pre_generate_images function here or from another file
from dataset import generate_random_phantom, create_incomplete_sinogram


def pre_generate_images(num_samples, size, missing_row_start, missing_row_end, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    angles = np.linspace(0., 180., size, endpoint=False)
    images_list = []

    save_images_dir = os.path.join(run_dir, "pre_generated_images")
    os.makedirs(save_images_dir, exist_ok=True)

    for i in range(num_samples):
        phantom = generate_random_phantom(size=size)
        complete_sino, incomplete_sino = create_incomplete_sinogram(
            phantom,
            angles=angles,
            missing_row_start=missing_row_start,
            missing_row_end=missing_row_end
        )
        images_list.append((incomplete_sino, complete_sino))

        # Save a small debug figure
        fig, axs = plt.subplots(1, 3, figsize=(9,3))
        axs[0].imshow(phantom, cmap='gray')
        axs[0].set_title("Phantom")
        axs[1].imshow(incomplete_sino, cmap='gray')
        axs[1].set_title("Incomplete Sinogram")
        axs[2].imshow(complete_sino, cmap='gray')
        axs[2].set_title("Complete Sinogram")
        for ax in axs: ax.axis('off')

        out_path = os.path.join(save_images_dir, f"sample_{i}.png")
        plt.savefig(out_path)
        plt.close(fig)

    return images_list

def main():
    # 2) Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train sinogram completion with optional pre-generation of data.")
    parser.add_argument('--pre_generate_data', action='store_true',
                        help='If set, pre-generate all images before training (instead of on-the-fly).')
    args = parser.parse_args()

    # 3) Create a unique timestamp for this training run
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("train_runs", timestamp_str)
    os.makedirs(run_dir, exist_ok=True)

    # 4) Setup logging in that run_dir
    logger = setup_logging(log_dir=run_dir)
    logger.info("=== Starting Main Script ===")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"CLI Options: pre_generate_data={args.pre_generate_data}")

    # 5) Hyperparameters
    IMG_SIZE = 128
    NUM_SAMPLES = 2000
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    ALPHA = 0.7
    SAVE_INTERVAL = 5
    MISSING_ROW_START = 40
    MISSING_ROW_END = 60

    # 6) If pre_generate_data, create images_list
    pre_generated_list = None
    if args.pre_generate_data:
        logger.info("Pre-generating data for training...")
        pre_generated_list = pre_generate_images(
            num_samples=NUM_SAMPLES,
            size=IMG_SIZE,
            missing_row_start=MISSING_ROW_START,
            missing_row_end=MISSING_ROW_END,
            run_dir=run_dir  # store images here
        )
        logger.info(f"Pre-generated {len(pre_generated_list)} samples saved under {run_dir}/pre_generated_images")

    # 7) Construct the dataset
    full_dataset = SinogramDataset(
        size=IMG_SIZE,
        num_samples=NUM_SAMPLES,
        missing_row_start=MISSING_ROW_START,
        missing_row_end=MISSING_ROW_END,
        pre_generated_data=pre_generated_list  # if None => old style
    )
    logger.info(f"Created dataset with {len(full_dataset)} samples")

    # 8) Split train/test
    train_subset, test_subset = split_train_test(full_dataset, train_ratio=0.8)
    logger.info(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")

    # 9) Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 10) Initialize model
    model = UNet(in_channels=1, out_channels=1)

    # 11) Train model; logs and checkpoints in run_dir
    train_losses, test_losses = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=LR,
        alpha=ALPHA,
        save_interval=SAVE_INTERVAL,
        run_dir=run_dir
    )

    # 12) Final sample test
    model.eval()
    with torch.no_grad():
        incomplete_sino, complete_sino = full_dataset[0]
        device = next(model.parameters()).device
        incomplete_sino = incomplete_sino.unsqueeze(0).to(device)
        output = model(incomplete_sino)

    incomplete_sino_np = incomplete_sino.squeeze().cpu().numpy()
    complete_sino_np = complete_sino.squeeze().numpy()
    output_np = output.squeeze().cpu().numpy()

    # 13) Save final reconstruction in run_dir
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(incomplete_sino_np, cmap='gray')
    ax[0].set_title("Incomplete Sinogram")
    ax[1].imshow(output_np, cmap='gray')
    ax[1].set_title("Predicted Complete")
    ax[2].imshow(complete_sino_np, cmap='gray')
    ax[2].set_title("Ground Truth Complete")

    final_fig_path = os.path.join(run_dir, "final_reconstruction.png")
    plt.savefig(final_fig_path)
    plt.close(fig)
    logger.info(f"Saved final reconstruction to {final_fig_path}")

    logger.info("=== End of Main Script ===")

if __name__ == "__main__":
    main()
