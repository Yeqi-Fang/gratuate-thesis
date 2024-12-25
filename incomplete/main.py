# main.py
import os
import logging
import datetime
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# local imports
from dataset import SinogramDataset
from dataset import generate_random_phantom, create_incomplete_sinogram  # if needed
from model import UNet
from train import train_model
from logger_utils import setup_logging

# If you prefer to directly import from generate_dataset:
from generate_dataset import generate_dataset


def pre_generate_images(num_samples, size, missing_row_start, missing_row_end, run_dir):
    """
    Generates all phantom images beforehand, saves them to disk (optional),
    and returns a list of (incomplete_sino, complete_sino) arrays in memory.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

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

        # Optional: Save a small debug figure for each sample
        fig, axs = plt.subplots(1, 3, figsize=(9,3))
        axs[0].imshow(phantom, cmap='gray')
        axs[0].set_title("Phantom")
        axs[1].imshow(incomplete_sino, cmap='gray')
        axs[1].set_title("Incomplete Sinogram")
        axs[2].imshow(complete_sino, cmap='gray')
        axs[2].set_title("Complete Sinogram")
        for ax in axs:
            ax.axis('off')

        out_path = os.path.join(save_images_dir, f"sample_{i}.png")
        plt.savefig(out_path)
        plt.close(fig)

    return images_list


def main():
    # 2) Parse command-line arguments
    parser = argparse.ArgumentParser(description="Main script for sinogram completion.")
    parser.add_argument('--data_mode', type=str, default='on_the_fly',
                        choices=['on_the_fly', 'existing', 'generate'],
                        help="How to obtain dataset: 'existing' folder, 'generate' beforehand, or 'on_the_fly' random.")
    parser.add_argument('--dataset_folder', type=str, default='my_dataset',
                        help="Folder containing or storing dataset .npy files (for 'existing' or 'generate' modes).")
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()

    # 1) Timestamped run folder
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("train_runs", timestamp_str)
    os.makedirs(run_dir, exist_ok=True)

    # 2) Setup logging
    logger = setup_logging(log_dir=run_dir)
    logger.info("=== Starting Main Script ===")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Command-line args: {vars(args)}")

    # 3) Setup dataset according to data_mode
    if args.data_mode == 'existing':
        # Use existing .npy dataset in args.dataset_folder
        dataset_folder = args.dataset_folder
        logger.info(f"Using existing dataset from {dataset_folder}")
        full_dataset = SinogramDataset(
            num_samples=args.num_samples,  # might not matter if folder has fewer/more
            size=args.img_size,
            dataset_folder=dataset_folder
        )
    elif args.data_mode == 'generate':
        # Generate dataset .npy files into dataset_folder, then load it
        dataset_folder = args.dataset_folder
        logger.info(f"Generating dataset of {args.num_samples} samples into {dataset_folder} ...")
        generate_dataset(
            num_samples=args.num_samples,
            size=args.img_size,
            missing_row_start=40,
            missing_row_end=60,
            out_folder=dataset_folder
        )
        logger.info("Dataset generated. Now loading from disk ...")
        # Now load the newly generated dataset
        full_dataset = SinogramDataset(
            num_samples=args.num_samples,
            size=args.img_size,
            dataset_folder=dataset_folder
        )
    else:
        # on_the_fly
        logger.info("Generating on-the-fly dataset (no prior saving).")
        full_dataset = SinogramDataset(
            num_samples=args.num_samples,
            size=args.img_size,
            dataset_folder=None  # triggers random generation
        )

    # 4) Split train/test
    from dataset import split_train_test
    train_subset, test_subset = split_train_test(full_dataset, train_ratio=0.8)
    logger.info(f"Train subset: {len(train_subset)}, Test subset: {len(test_subset)}")

    # 8) Split train/test
    train_subset, test_subset = split_train_test(full_dataset, train_ratio=0.8)
    logger.info(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")

    # 5) Create Dataloaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 6) Initialize Model
    model = UNet(in_channels=1, out_channels=1)

    # 7) Train
    train_losses, test_losses = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        device='cuda' if torch.cuda.is_available() else "cpu",
        save_interval=args.save_interval,
        run_dir=run_dir
    )

    # 8) Test on a single sample
    model.eval()
    with torch.no_grad():
        incomplete_sino, complete_sino = full_dataset[0]
        device = next(model.parameters()).device
        incomplete_sino = incomplete_sino.unsqueeze(0).to(device)
        output = model(incomplete_sino)

    inc_np = incomplete_sino.squeeze().cpu().numpy()
    com_np = complete_sino.squeeze().numpy()
    out_np = output.squeeze().cpu().numpy()

    # 9) Save final example
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(inc_np, cmap='gray')
    axs[0].set_title("Incomplete Sinogram")
    axs[1].imshow(out_np, cmap='gray')
    axs[1].set_title("Predicted Complete")
    axs[2].imshow(com_np, cmap='gray')
    axs[2].set_title("Ground Truth Complete")

    final_img = os.path.join(run_dir, "final_reconstruction.png")
    plt.savefig(final_img)
    plt.close(fig)
    logger.info(f"Saved final reconstruction to {final_img}")

    logger.info("=== End of Main Script ===")

if __name__ == "__main__":
    main()
