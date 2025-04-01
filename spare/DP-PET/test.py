import torch
import matplotlib.pyplot as plt

def test_and_visualize(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for sinogram, image in test_loader:
            sinogram, image = sinogram.to(device), image.to(device)
            reconstructed = model(sinogram)
            break

    # Reshape for visualization
    original_image = image[0].cpu().numpy().squeeze()
    reconstructed_image = reconstructed[0].cpu().numpy().squeeze()

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
