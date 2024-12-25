# main.py
import os
import logging
import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

from dataset import SinogramDataset, split_train_test
from model import UNet
from train import train_model
from logger_utils import setup_logging

def main():
    # 1) Create a unique timestamp for this training run
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("train_runs", timestamp_str)
    os.makedirs(run_dir, exist_ok=True)

    # 2) Setup logging in that run_dir
    logger = setup_logging(log_dir=run_dir)
    logger.info("=== Starting Main Script ===")
    logger.info(f"Run directory: {run_dir}")

    # 3) Hyperparameters
    IMG_SIZE = 128
    NUM_SAMPLES = 200
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 5e-3
    ALPHA = 0.5
    SAVE_INTERVAL = 5

    # 4) Create dataset
    full_dataset = SinogramDataset(
        size=IMG_SIZE,
        num_samples=NUM_SAMPLES,
        missing_row_start=40,
        missing_row_end=60
    )
    logger.info(f"Created dataset with {len(full_dataset)} samples")

    # 5) Split train/test
    train_subset, test_subset = split_train_test(full_dataset, train_ratio=0.8)
    logger.info(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")

    # 6) Create DataLoaders
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

    # 7) Initialize model
    model = UNet(in_channels=1, out_channels=1)

    # 8) Train model; all logs and checkpoints go in run_dir
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

    # 9) Final sample test
    model.eval()
    with torch.no_grad():
        incomplete_sino, complete_sino = full_dataset[0]
        device = next(model.parameters()).device
        incomplete_sino = incomplete_sino.unsqueeze(0).to(device)
        output = model(incomplete_sino)

    incomplete_sino_np = incomplete_sino.squeeze().cpu().numpy()
    complete_sino_np = complete_sino.squeeze().numpy()
    output_np = output.squeeze().cpu().numpy()

    # 10) Save final reconstruction in run_dir
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
