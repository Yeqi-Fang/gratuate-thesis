from data_preparation import prepare_dataloaders
from train import train_and_save_best_model
from test import test_and_visualize
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_dataloaders(size=64, samples=1000, batch_size=128)

    # Train and save the best model
    model = train_and_save_best_model(train_loader, test_loader, device, epochs=10, save_path='best_model.pth')

    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Test and visualize results
    test_and_visualize(model, test_loader, device)
