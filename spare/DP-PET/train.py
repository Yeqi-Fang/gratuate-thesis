import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import ImprovedUNetReconstructor
import matplotlib.pyplot as plt


def train_and_save_best_model(train_loader, test_loader, device, epochs=200, save_path='best_model.pth'):
    """
    Train the model, evaluate train/test loss after each epoch, and save the best model based on test loss.
    """
    model = ImprovedUNetReconstructor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')  # Initialize with a very high value
    best_model_state = None  # Store the best model's state_dict

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for sinogram, image in train_loader:
            sinogram, image = sinogram.to(device), image.to(device)
            optimizer.zero_grad()
            outputs = model(sinogram)
            loss = criterion(outputs, image)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # Evaluation phase
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for sinogram, image in test_loader:
                sinogram, image = sinogram.to(device), image.to(device)
                outputs = model(sinogram)
                loss = criterion(outputs, image)
                running_test_loss += loss.item()

        # Average losses
        train_loss = running_train_loss / len(train_loader)
        test_loss = running_test_loss / len(test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()  # Save best state_dict
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best Model Saved at Epoch {epoch+1} with Test Loss: {test_loss:.4f}")

        scheduler.step()  # Adjust the learning rate

    # Ensure the best model state is loaded before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"🎯 Loaded Best Model with Test Loss: {best_test_loss:.4f}")
    else:
        print("⚠️ No improvement observed during training. Returning last model state.")

    # Plot train and test losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model
