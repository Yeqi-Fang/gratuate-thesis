import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

i = 6

# Paths to the images.
orig_path = rf"C:\Users\fangy\Desktop\reconstructed\reconstructed_index{i}_num2000000000.npy"
recon_path = rf"C:\Users\fangy\Desktop\20250327_182411\reconstructed_from_sinogram_incomplete_{i+1}.npy"

# Load the images.
orig_image = np.load(orig_path)
recon_image = np.load(recon_path).transpose(2, 1, 0)

print(orig_image.max(), orig_image.min())
print(recon_image.max(), recon_image.min())

# Print image shapes for verification.
print("Original image shape:", orig_image.shape)
print("Reconstructed image shape:", recon_image.shape)

# Select a slice index (middle slice along the z-axis).
slice_index = orig_image.shape[2] // 2

# Create a figure with two subplots for side-by-side comparison.
fig, axs = plt.subplots(1, 2, figsize=(12, 4.))

# Display the original image slice with colorbar.
im0 = axs[0].imshow(orig_image[:, :, slice_index], cmap='magma', interpolation='nearest')
axs[0].set_title(f"Original Image (Slice {slice_index})")
axs[0].axis('off')
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

# Display the reconstructed image slice with colorbar.
im1 = axs[1].imshow(recon_image[:, :, slice_index], cmap='magma', interpolation='nearest')
# im1 = axs[1].imshow(recon_image[:, slice_index, :], cmap='magma', interpolation='nearest')
# im1 = axs[1].imshow(recon_image[slice_index, :, :], cmap='magma', interpolation='nearest')
axs[1].set_title(f"Reconstructed Image (Slice {slice_index})")
axs[1].axis('off')
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("paper_image/compare_reconstruction_restoration.pdf")
# plt.show()


print("PSNR:", psnr(orig_image, recon_image))
print("SSIM:", ssim(orig_image, recon_image, data_range=orig_image.max() - orig_image.min()))