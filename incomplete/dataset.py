# dataset.py
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
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


def create_incomplete_sinogram(image, angles=None, missing_angle_start=40, missing_angle_end=60):
    """Compute radon transform and zero out a horizontal band."""
    if angles is None:
        angles = np.linspace(0., 180., max(image.shape), endpoint=False)
    print(angles)
    sinogram = radon(image, theta=angles, circle=False)
    mask = (angles >= missing_angle_start) & (angles < missing_angle_end)
    incomplete = sinogram.copy()
    incomplete[mask, :] = 0.0
    return sinogram, incomplete


class SinogramDataset(Dataset):
    """
    If dataset_folder is provided, we load .npy files from that folder.
    Else, we generate on-the-fly (generate_random_phantom).
    """
    def __init__(self, 
                 num_samples=100, 
                 size=128, 
                 missing_angle_start=40, 
                 missing_angle_end=60, 
                 dataset_folder=None):
        super().__init__()
        self.num_samples = num_samples
        self.size = size
        self.missing_angle_start = missing_angle_start
        self.missing_angle_end = missing_angle_end

        self.dataset_folder = dataset_folder
        if self.dataset_folder is not None:
            # mode: use existing dataset
            # find all incomplete_x.npy, complete_x.npy
            self.incomplete_files = []
            self.complete_files = []
            for fname in os.listdir(dataset_folder):
                if fname.startswith("incomplete_") and fname.endswith(".npy"):
                    idx = int(fname.split('_')[1].split('.')[0])  # e.g. incomplete_0.npy -> 0
                    self.incomplete_files.append((idx, os.path.join(dataset_folder, fname)))
                elif fname.startswith("complete_") and fname.endswith(".npy"):
                    idx = int(fname.split('_')[1].split('.')[0])
                    self.complete_files.append((idx, os.path.join(dataset_folder, fname)))

            # sort by idx to ensure consistent order
            self.incomplete_files.sort(key=lambda x: x[0])
            self.complete_files.sort(key=lambda x: x[0])

            # check length consistency
            if len(self.incomplete_files) != len(self.complete_files):
                raise ValueError("Mismatch between incomplete and complete files in dataset folder!")

            self.num_samples = len(self.incomplete_files)  # override
        else:
            # mode: generate on-the-fly
            # angles for radon
            self.angles = np.linspace(0., 180., self.size, endpoint=False)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.dataset_folder is not None:
            # load from .npy files
            inc_idx, inc_path = self.incomplete_files[idx]
            comp_idx, comp_path = self.complete_files[idx]
            # assert inc_idx == comp_idx if needed
            incomplete_sino = np.load(inc_path)
            complete_sino = np.load(comp_path)
        else:
            # On-the-fly generation
            phantom = generate_random_phantom(size=self.size)
            complete_sino, incomplete_sino = create_incomplete_sinogram(
                phantom,
                angles=self.angles,
                missing_angle_start=self.missing_angle_start,
                missing_angle_end=self.missing_angle_end
            )
        # Convert to torch tensor
        incomplete_t = torch.tensor(incomplete_sino, dtype=torch.float32).unsqueeze(0)
        complete_t = torch.tensor(complete_sino, dtype=torch.float32).unsqueeze(0)
        return incomplete_t, complete_t

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
