# train.py
import os
import torch
import logging
from losses import CombinedLoss
from torch import optim

def save_checkpoint(model, epoch, optimizer=None, save_dir='saved_models'):
    """
    Saves a checkpoint with model state (and optionally optimizer state).
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"[Checkpoint] Saved: {checkpoint_path}")

def train_model(
    model,
    train_loader,
    test_loader,
    epochs=5,
    lr=1e-3,
    alpha=0.5,
    save_interval=5,
    save_dir='saved_models'
):
    """
    Train and validate the model, tracking both training and validation losses per epoch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CombinedLoss(alpha=alpha)

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs+1):
        ### TRAIN ###
        model.train()
        train_loss = 0.0
        for incomplete_sino, complete_sino in train_loader:
            incomplete_sino = incomplete_sino.to(device)
            complete_sino = complete_sino.to(device)

            optimizer.zero_grad()
            output = model(incomplete_sino)
            loss = criterion(output, complete_sino)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        ### VALIDATION ###
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for incomplete_sino, complete_sino in test_loader:
                incomplete_sino = incomplete_sino.to(device)
                complete_sino = complete_sino.to(device)

                output = model(incomplete_sino)
                loss = criterion(output, complete_sino)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        test_losses.append(avg_val_loss)

        logging.info(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss:.5f} | Test Loss: {avg_val_loss:.5f}")

        # Optional: Save checkpoint periodically
        if epoch % save_interval == 0:
            save_checkpoint(model, epoch, optimizer=optimizer, save_dir=save_dir)

    return train_losses, test_losses
