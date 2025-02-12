# generate_reconstruct.py

import os
import re
import glob
import numpy as np
import torch
import time
from datetime import datetime
import matplotlib.pyplot as plt
import multiprocessing
from pytomography.io.PET import gate, shared

from pet_simulator.geometry import create_pet_geometry
from pet_simulator.simulator import PETSimulator
from pet_simulator.utils import save_events, save_events_binary
from reconstruction_all import reconstruct_volume_for_lmf
from outlier_detection import (
    global_outlier_detection,
    local_outlier_detection,
    edge_outlier_detection,
    combined_outlier_detection,
    analyze_outlier_masks,
    remove_outliers_iteratively
)

# PET scanner configuration information
info = {
    'min_rsector_difference': np.float32(0.0),
    'crystal_length': np.float32(0.0),
    'radius': np.float32(253.71),
    'crystalTransNr': 13,
    'crystalTransSpacing': np.float32(4.01648),
    'crystalAxialNr': 7,
    'crystalAxialSpacing': np.float32(5.36556),
    'submoduleAxialNr': 1,
    'submoduleAxialSpacing': np.float32(0.0),
    'submoduleTransNr': 1,
    'submoduleTransSpacing': np.float32(0.0),
    'moduleTransNr': 1,
    'moduleTransSpacing': np.float32(0.0),
    'moduleAxialNr': 6,
    'moduleAxialSpacing': np.float32(37.55892),
    'rsectorTransNr': 28,
    'rsectorAxialNr': 1,
    'TOF': 0,
    'NrCrystalsPerRing': 364, #13 * 7 * 4
    'NrRings': 42,
    'firstCrystalAxis': 0
}

# Reconstruction parameters
voxel_size = 2.78
volume_size = 128
extended_size = 128
n_iters = 1
n_subsets = 34
psf_fwhm_mm = 4.5
outlier = False
num_events = int(1.5e9)
save_events_pos = False

def main():
    # Create PET scanner geometry from info
    geometry = create_pet_geometry(info)

    # Save the detector lookup table once.
    # (Since LUT depends only on geometry, a dummy image is sufficient.)
    dummy_image = np.ones((1,1,1), dtype=np.float32)
    simulator_for_lut = PETSimulator(geometry, dummy_image, voxel_size=2.78)
    simulator_for_lut.save_detector_positions("detector_lut.txt")

    # Define the base directory where the 3D image files are stored.
    base_dir = r"D:\Datasets\dataset\train_npy_crop"
    lut_file = 'detector_lut.txt'
    
    # Create an output directory for listmode data if it doesn't exist.
    lmf_output_dir = "tmp"
    output_dir = f'reconstruction_npy_full_train/{num_events:d}'
    os.makedirs(lmf_output_dir, exist_ok=True)
    
    # Create a log directory using the current timestamp
    log_dir = os.path.join("log", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=False)
    print(f"Log directory: {log_dir}")
    
    output_dir_sinogram = f'reconstruction_npy_full_train/{num_events:d}/sinogram'
    os.makedirs(output_dir_sinogram, exist_ok=True)

    # Process each image file from 3d_image_0.npy to 3d_image_169.npy
    for i in range(97, 170):
        image_filename = f"3d_image_{i}.npy"
        image_path = os.path.join(base_dir, image_filename)
        print(f"\nProcessing {image_filename} ...")
        
        # Load the 3D image (which acts as the probability density distribution)
        image = np.load(image_path)
        print("Image shape:", image.shape)
        
        # Create the simulator with the current image and voxel size
        simulator = PETSimulator(geometry, image, voxel_size=2.78, save_events_pos=save_events_pos)
        
        # Run the simulation
        print("Starting simulation...")
        start_time = time.time()
        events = simulator.simulate_events(num_events=num_events, use_multiprocessing=True)
        end_time = time.time()
        print(f"Generated {len(events)} valid events in {end_time - start_time:.2f} seconds.")
        print("Events shape:", events.shape)
        
        # Save events in binary .lmf format (minimal and full)
        minimal_file = os.path.join(lmf_output_dir, f"listmode_data_minimal_{i}_{num_events}")
        save_events_binary(minimal_file, events, save_full_data=False)

        lmf_path = minimal_file + ".npz"
        filename_regex = re.compile(r".*listmode_data_minimal_(\d+)_(\d+)\.npz")
        print(f"\nProcessing {lmf_path}")
        start_time = time.time()
        
        # base_name = os.path.basename(lmf_output_dir)
        match = filename_regex.match(lmf_path)
        if match:
            index_str, num_str = match.groups()
            out_name = f"reconstructed_index{index_str}_num{num_str}"
        else:
            raise ValueError(f"Filename {lmf_path} does not match the expected format.")

        # Reconstruct
        result_3d = reconstruct_volume_for_lmf(
            lmf_file=lmf_path,
            lut_file=lut_file,
            voxel_size=voxel_size,
            volume_size=volume_size,
            extended_size=extended_size,
            n_iters=n_iters,
            n_subsets=n_subsets,
            psf_fwhm_mm=psf_fwhm_mm,
            detector_outlier=outlier
        )

        # Save
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, result_3d)
        
        os.remove(lmf_path)
        end_time = time.time()

        print(f"  -> Saved {out_path} (shape={result_3d.shape}) in {end_time - start_time:.1f}s")

        detector_ids = torch.from_numpy(events).to(torch.int32)
        del events, simulator
        sinogram_randoms_true = gate.listmode_to_sinogram(detector_ids, info)
        del detector_ids
        print("Sinogram shape:", sinogram_randoms_true.shape)
        
        # visualize sinogram
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        im0 = ax.imshow(sinogram_randoms_true.numpy()[0, :, :42], cmap='magma')
        ax.set_title(f'Original Sinogram')
        ax.axis('off')
        fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig_filename = os.path.join(log_dir, f"sinogram_{image_filename}.pdf")
        plt.savefig(fig_filename, dpi=300)
        plt.close(fig)
        print(f"  -> Saved sinogram figure to {fig_filename}")
                
        out_path_sinogram = os.path.join(output_dir_sinogram, out_name)
        np.save(out_path_sinogram, sinogram_randoms_true.numpy().astype(np.float32))
        del sinogram_randoms_true
        
        # ---------------------------
        # LOGGING: Save comparison figures
        # ---------------------------
        # We want to compare a slice of the original image and the reconstructed volume.
        # Here we choose the middle slice along the z-axis.
        slice_index = result_3d.shape[2] // 2
        
        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 4.5))
        im0 = axs[0].imshow(image[:, :, slice_index], cmap='magma', interpolation='nearest')
        axs[0].set_title(f'Original Image Slice (z = {slice_index})')
        axs[0].axis('off')
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        
        im1 = axs[1].imshow(result_3d[:, :, slice_index], cmap='magma', interpolation='nearest')
        axs[1].set_title(f'Reconstructed Slice (z = {slice_index})')
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        # Save the figure in the log directory
        fig_filename = os.path.join(log_dir, f"comparison_{image_filename}.pdf")
        plt.savefig(fig_filename, dpi=300)
        plt.close(fig)
        print(f"  -> Saved comparison figure to {fig_filename}")
        
        # Optionally, you can save a second figure (e.g., for a slice along x-axis)
        slice_index_x = result_3d.shape[0] // 2
        fig2, axs2 = plt.subplots(1, 2, figsize=(15, 6))
        im0 = axs2[0].imshow(image[slice_index_x, :, :], cmap='magma', interpolation='nearest')
        axs2[0].set_title(f'Original Image Slice (x = {slice_index_x})')
        axs2[0].axis('off')
        fig.colorbar(im0, ax=axs2[0], fraction=0.046, pad=0.04)
        
        im1 = axs2[1].imshow(result_3d[slice_index_x, :, :], cmap='magma', interpolation='nearest')
        axs2[1].set_title(f'Reconstructed Slice (x = {slice_index_x})')
        axs2[1].axis('off')
        fig.colorbar(im1, ax=axs2[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig2_filename = os.path.join(log_dir, f"comparison_x_{image_filename}.pdf")
        plt.savefig(fig2_filename, dpi=300)
        plt.close(fig2)
        print(f"  -> Saved second comparison figure to {fig2_filename}")
        
        # Optionally, you can save a third figure (e.g., for a slice along y-axis)
        slice_index_x = result_3d.shape[1] // 2
        fig2, axs3 = plt.subplots(1, 2, figsize=(15, 4.5))
        im0 = axs3[0].imshow(image[:, slice_index_x, :], cmap='magma', interpolation='nearest')
        axs3[0].set_title(f'Original Image Slice (x = {slice_index_x})')
        axs3[0].axis('off')
        fig.colorbar(im0, ax=axs3[0], fraction=0.046, pad=0.04)
        
        im1 = axs3[1].imshow(result_3d[:, slice_index_x, :], cmap='magma', interpolation='nearest')
        axs3[1].set_title(f'Reconstructed Slice (x = {slice_index_x})')
        axs3[1].axis('off')
        fig.colorbar(im1, ax=axs3[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig3_filename = os.path.join(log_dir, f"comparison_y_{image_filename}.pdf")
        plt.savefig(fig3_filename, dpi=300)
        plt.close(fig2)
        print(f"  -> Saved second comparison figure to {fig3_filename}")

    print("\nAll reconstructions complete.")
    
if __name__ == "__main__":
    main()