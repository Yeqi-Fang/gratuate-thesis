import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

i = 22

# Paths to the images.
orig_dir = rf"C:\Users\fangy\Desktop\reconstructed_rm_padded_1e-8"
recon_dir = rf"C:\Users\fangy\Desktop\all_prediction_merged_reconstructed\all_prediction_merged_reconstructed_rm_padded_1e-8_rotated"

file_list = os.listdir(orig_dir)

psnr_list = []
ssim_list = []

for i in file_list:
    if i.endswith(".npy"):
        recon_file = os.path.join(recon_dir, i)
        orig_file = os.path.join(orig_dir, i)
        # Load the images.
        orig_image = np.load(orig_file)
        recon_image = np.load(recon_file)
        # calc psnr and ssim
        recon_image = np.flip(recon_image, axis=2)

        psnr_value = psnr(orig_image, recon_image)
        ssim_value = ssim(orig_image, recon_image, data_range=orig_image.max() - orig_image.min())
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        

# find mean and histgram

mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)

print("Mean PSNR:", mean_psnr)
print("Mean SSIM:", mean_ssim)

# # Print image shapes for verification.

# plot histogram

plt.figure(figsize=(12, 4.))
plt.subplot(1, 2, 1)
plt.hist(psnr_list, bins=50, color='blue', alpha=0.7)
plt.title("PSNR Histogram")
plt.xlabel("PSNR")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(ssim_list, bins=50, color='green', alpha=0.7)
plt.title("SSIM Histogram")
plt.xlabel("SSIM")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("paper_image/compare_reconstruction_restoration_histogram.pdf")
plt.show()

# print("PSNR:", psnr(orig_image, recon_image))
# print("SSIM:", ssim(orig_image, recon_image, data_range=orig_image.max() - orig_image.min()))