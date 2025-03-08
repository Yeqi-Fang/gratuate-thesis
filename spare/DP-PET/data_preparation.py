import numpy as np
import torch
from skimage.draw import disk, ellipse
from skimage.transform import radon, resize
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import random


def generate_random_image(size=128, num_shapes=5):
    """
    Generate a random image with multiple circles and ellipses.
    """
    image = np.zeros((size, size), dtype=np.float32)

    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'ellipse'])
        center_x = random.randint(0, size - 1)
        center_y = random.randint(0, size - 1)
        intensity = random.uniform(0.3, 1.0)

        if shape_type == 'circle':
            radius = random.randint(size // 10, size // 4)
            rr, cc = disk((center_x, center_y), radius, shape=image.shape)
        else:  # Ellipse
            radius_x = random.randint(size // 10, size // 4)
            radius_y = random.randint(size // 10, size // 4)
            rotation = random.uniform(0, 360)
            rr, cc = ellipse(center_x, center_y, radius_x, radius_y, shape=image.shape, rotation=np.deg2rad(rotation))

        image[rr, cc] = intensity

    image = np.clip(image, 0, 1)
    return image


def generate_incomplete_sinogram(image, size=128, missing_ranges=[(0, 30), (60, 90)]):
    """
    Generate incomplete sinogram by masking specific angle ranges.
    """
    angle_steps = np.linspace(0., 180., size, endpoint=False)
    sinogram = radon(image, theta=angle_steps, circle=True)

    # Create a consistent mask for the specified angle ranges
    mask = np.ones(size, dtype=bool)
    for start, end in missing_ranges:
        mask[(angle_steps >= start) & (angle_steps < end)] = False

    incomplete_sinogram = sinogram.copy()
    incomplete_sinogram[:, ~mask] = 0  # Apply the mask

    return incomplete_sinogram, sinogram


def generate_data(size=128, samples=500, num_shapes=5, missing_ranges=[(0, 30), (60, 90)]):
    """
    Generate a dataset of incomplete and complete sinograms.
    """
    incomplete_sinograms = []
    complete_sinograms = []
    for _ in range(samples):
        image = generate_random_image(size=size, num_shapes=num_shapes)
        incomplete_sinogram, complete_sinogram = generate_incomplete_sinogram(
            image, size=size, missing_ranges=missing_ranges)
        
        incomplete_sinograms.append(incomplete_sinogram)
        complete_sinograms.append(complete_sinogram)
    
    return np.array(incomplete_sinograms), np.array(complete_sinograms)


def prepare_dataloaders(size=128, samples=500, batch_size=16, missing_ranges=[(0, 30), (60, 90)]):
    """
    Prepare PyTorch DataLoaders for incomplete-to-complete sinogram dataset.
    """
    incomplete_sinograms, complete_sinograms = generate_data(
        size=size, samples=samples, missing_ranges=missing_ranges)
    incomplete_sinograms = incomplete_sinograms[..., np.newaxis]  # Add channel dimension
    complete_sinograms = complete_sinograms[..., np.newaxis]  # Add channel dimension

    X_train, X_test, y_train, y_test = train_test_split(incomplete_sinograms, complete_sinograms, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2),
        torch.tensor(y_train, dtype=torch.float32).permute(0, 3, 1, 2)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2),
        torch.tensor(y_test, dtype=torch.float32).permute(0, 3, 1, 2)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def visualize_data_samples(incomplete_sinograms, complete_sinograms, num_samples=5):
    """
    Visualize a few incomplete and complete sinogram pairs.
    """
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.title(f"Incomplete Sinogram {i+1}")
        plt.imshow(incomplete_sinograms[i].squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.title(f"Complete Sinogram {i+1}")
        plt.imshow(complete_sinograms[i].squeeze(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test dataset generation
    size = 128
    samples = 5
    missing_ranges = [(10, 30), (130, 160)]

    incomplete_sinograms, complete_sinograms = generate_data(
        size=size, samples=samples, missing_ranges=missing_ranges)
    visualize_data_samples(incomplete_sinograms, complete_sinograms, num_samples=5)
