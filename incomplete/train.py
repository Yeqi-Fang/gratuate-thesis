# train.py
import os
import logging
import torch
import matplotlib.pyplot as plt
from torch import optim
from losses import CombinedLoss

def save_checkpoint(model, epoch, optimizer, save_dir):
    """
    Saves the model checkpoint into the 'save_dir' directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"[Checkpoint] Saved model to: {checkpoint_path}")


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=5,
    lr=1e-3,
    alpha=0.5,
    device=None,
    save_interval=5,
    run_dir="."
):
    """
    Train and validate the model, storing:
      - logs in run_dir (already set by logger).
      - checkpoints in run_dir (timestamped folder).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CombinedLoss(alpha=alpha)

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs+1):
        # Train
        model.train()
        train_loss_sum = 0.0
        for incomplete_sino, complete_sino in train_loader:
            incomplete_sino = incomplete_sino.to(device)
            complete_sino = complete_sino.to(device)

            optimizer.zero_grad()
            output = model(incomplete_sino)
            loss = criterion(output, complete_sino)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        train_avg = train_loss_sum / len(train_loader)
        train_losses.append(train_avg)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for incomplete_sino, complete_sino in test_loader:
                incomplete_sino = incomplete_sino.to(device)
                complete_sino = complete_sino.to(device)
                output = model(incomplete_sino)
                loss = criterion(output, complete_sino)
                val_loss_sum += loss.item()
        val_avg = val_loss_sum / len(test_loader) if len(test_loader) > 0 else 0
        test_losses.append(val_avg)

        logging.info(f"Epoch[{epoch}/{epochs}] Train Loss: {train_avg:.5f}, Test Loss: {val_avg:.5f}")

        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(run_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    # Optionally, you can also save final training curves, etc., inside `run_dir`
    # For example:
    fig = plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train vs Test Loss')
    out_fig_path = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(out_fig_path)
    plt.close(fig)
    logging.info(f"Saved loss curve to: {out_fig_path}")

    return train_losses, test_losses
