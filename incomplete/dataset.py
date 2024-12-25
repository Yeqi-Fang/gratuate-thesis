# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.draw import disk, ellipse
from skimage.transform import radon

def generate_random_phantom(size=128, max_shapes=5):
    """
    Generate a random phantom image of shape [size, size].
    Creates circles/ellipses emphasizing horizontal alignment.
    """
    image = np.zeros((size, size), dtype=np.float32)
    num_shapes = np.random.randint(1, max_shapes + 1)
    for _ in range(num_shapes):
        shape_type = np.random.choice(["circle", "ellipse"])
        center_x = np.random.randint(size // 4, 3 * size // 4)  # horizontal restriction
        center_y = np.random.randint(0, size)                   # full vertical
        radius_x = np.random.randint(5, size // 6)
        radius_y = np.random.randint(size // 6, size // 3)
        intensity = np.random.uniform(0.5, 1.0)
        
        if shape_type == "circle":
            rr, cc = disk((center_y, center_x), radius_y)
            rr = np.clip(rr, 0, size - 1)
            cc = np.clip(cc, 0, size - 1)
            image[rr, cc] += intensity
        else:
            rr, cc = ellipse(center_y, center_x, radius_y, radius_x, shape=image.shape)
            image[rr, cc] += intensity
    
    return np.clip(image, 0, 1)


def create_incomplete_sinogram(image, angles=None, missing_row_start=40, missing_row_end=60):
    """
    Perform Radon transform to create a sinogram, then zero out certain rows
    to simulate a horizontally missing band.
    """
    if angles is None:
        angles = np.linspace(0., 180., max(image.shape), endpoint=False)
    
    sinogram = radon(image, theta=angles, circle=False)
    sinogram = sinogram.T  # transpose for horizontal alignment

    incomplete = sinogram.copy()
    incomplete[missing_row_start:missing_row_end, :] = 0.0
    return sinogram, incomplete


class SinogramDataset(Dataset):
    """
    PyTorch Dataset for generating random phantoms and sinograms on-the-fly.
    """
    def __init__(self, size=128, num_samples=1000, missing_row_start=40, missing_row_end=60):
        super().__init__()
        self.size = size
        self.num_samples = num_samples
        self.missing_row_start = missing_row_start
        self.missing_row_end = missing_row_end
        self.angles = np.linspace(0., 180., self.size, endpoint=False)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        phantom = generate_random_phantom(self.size)
        complete_sino, incomplete_sino = create_incomplete_sinogram(
            phantom,
            angles=self.angles,
            missing_row_start=self.missing_row_start,
            missing_row_end=self.missing_row_end
        )
        
        # Convert to [1, H, W] PyTorch tensors
        complete_sino_t = torch.tensor(complete_sino, dtype=torch.float32).unsqueeze(0)
        incomplete_sino_t = torch.tensor(incomplete_sino, dtype=torch.float32).unsqueeze(0)
        return incomplete_sino_t, complete_sino_t


def split_train_test(dataset, train_ratio=0.8):
    """
    Splits dataset into two subsets: train & test.
    """
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    test_size = total_samples - train_size
    
    indices = torch.randperm(total_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    return train_subset, test_subset
