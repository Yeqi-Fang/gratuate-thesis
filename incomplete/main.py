# main.py
import logging
import matplotlib.pyplot as plt
import torch

from dataset import SinogramDataset, split_train_test
from model import UNet
from train import train_model
from logger_utils import setup_logging

def main():
    # 1) Setup logging
    logger = setup_logging(log_dir='logs')
    logger.info("=== Starting Main Script ===")

    # 2) Hyperparameters
    IMG_SIZE = 128
    NUM_SAMPLES = 200
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 5e-3
    ALPHA = 0.5  # weighting for combined loss

    # 3) Create Single Full Dataset
    full_dataset = SinogramDataset(
        size=IMG_SIZE,
        num_samples=NUM_SAMPLES,
        missing_row_start=40,
        missing_row_end=60
    )
    logger.info(f"Created dataset with {len(full_dataset)} samples.")

    # 4) Split into Train & Test
    train_subset, test_subset = split_train_test(full_dataset, train_ratio=0.8)
    logger.info(f"Train size: {len(train_subset)}, Test size: {len(test_subset)}")

    # 5) Create DataLoaders
    from torch.utils.data import DataLoader
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

    # 6) Initialize Model
    model = UNet(in_channels=1, out_channels=1)

    # 7) Train Model
    train_losses, test_losses = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=LR,
        alpha=ALPHA,
        save_interval=5,       # Save every 5 epochs
        save_dir='saved_models'
    )

    # 8) Plot Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Train vs. Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 9) Test on a Random Sample
    model.eval()
    with torch.no_grad():
        incomplete_sino, complete_sino = full_dataset[0]
        device = next(model.parameters()).device
        incomplete_sino = incomplete_sino.unsqueeze(0).to(device)
        output = model(incomplete_sino)

    incomplete_sino_np = incomplete_sino.squeeze().cpu().numpy()
    complete_sino_np = complete_sino.squeeze().numpy()
    output_np = output.squeeze().cpu().numpy()

    # 10) Visualize Single Sample
    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    ax[0].imshow(incomplete_sino_np, cmap='gray')
    ax[0].set_title("Incomplete Sinogram")
    ax[1].imshow(output_np, cmap='gray')
    ax[1].set_title("Predicted Complete")
    ax[2].imshow(complete_sino_np, cmap='gray')
    ax[2].set_title("Ground Truth Complete")
    plt.show()

    logger.info("=== End of Main Script ===")


if __name__ == "__main__":
    main()
