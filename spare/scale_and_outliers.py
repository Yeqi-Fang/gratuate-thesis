import os
import numpy as np


from outlier_detection import (
    global_outlier_detection,
    local_outlier_detection,
    edge_outlier_detection,
    combined_outlier_detection,
    analyze_outlier_masks,
    remove_outliers_iteratively
)


# Define input and output folders.
input_folder = r"reconstruction_npy_full_train/1000000000_80_128_128"
output_folder = r"reconstruction_npy_full_train/1000000000_80_128_128/filtered"
os.makedirs(output_folder, exist_ok=True)

# Loop through all .npy files in the input folder.
for filename in os.listdir(input_folder):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_folder, filename)
        # Load the reconstructed volume.
        recon_image = np.load(file_path)
        
        # Scale the image to the [0, 1] range.
        # If the image has constant value, leave it as is.
        min_val = recon_image.min()
        max_val = recon_image.max()
        if max_val - min_val != 0:
            recon_image_scaled = (recon_image - min_val) / (max_val - min_val)
        else:
            recon_image_scaled = recon_image.copy()
        
        cleaned_image = remove_outliers_iteratively(
            recon_image_scaled,
            max_iters=10,
            global_factor=1.8, 
            global_percentile=99.0,
            local_window=3,
            local_sigma=3.0
        )
        # Save the scaled image to the output folder.
        output_path = os.path.join(output_folder, filename)
        np.save(output_path, cleaned_image)
        print(f"Processed {filename}: min={min_val:.4f}, max={max_val:.4f} -> saved scaled version.")
