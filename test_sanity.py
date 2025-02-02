import numpy as np
import matplotlib.pyplot as plt

# Paths to the images.
orig_path = r"D:\Datasets\dataset\train_npy\3d_image_8.npy"
recon_path = r"reconstruction_npy_full_train_scaled\reconstructed_8_128x128x128.npy"

# Load the images.
orig_image = np.load(orig_path)
recon_image = np.load(recon_path)

print(orig_image.max(), orig_image.min())
print(recon_image.max(), recon_image.min())

# Print image shapes for verification.
print("Original image shape:", orig_image.shape)
print("Reconstructed image shape:", recon_image.shape)

# Select a slice index (middle slice along the z-axis).
slice_index = orig_image.shape[2] // 2 + 60

# Create a figure with two subplots for side-by-side comparison.
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image slice with colorbar.
im0 = axs[0].imshow(orig_image[slice_index, :, :], cmap='magma', interpolation='nearest')
axs[0].set_title(f"Original Image (Slice {slice_index})")
axs[0].axis('off')
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

# Display the reconstructed image slice with colorbar.
im1 = axs[1].imshow(recon_image[slice_index, :, :], cmap='magma', interpolation='nearest')
axs[1].set_title(f"Reconstructed Image (Slice {slice_index})")
axs[1].axis('off')
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
