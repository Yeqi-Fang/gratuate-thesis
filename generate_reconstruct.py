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

import threading

def save_events_background(file_path, events, save_full_data=False):
    """Save events in a background thread to avoid blocking the main process."""
    thread = threading.Thread(
        target=save_events_binary,
        args=(file_path, events, save_full_data)
    )
    thread.daemon = False  # Ensure thread doesn't exit when main program exits
    thread.start()
    return thread




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
num_events = int(2e9)
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
    # base_dir = r"D:\Datasets\dataset\train_npy_crop"
    base_dir = "data/dataset/train_npy_crop"
    lut_file = 'detector_lut.txt'
    
    # Create and cache the scanner LUT for reuse
    lut_data = np.loadtxt(lut_file, skiprows=1)
    scanner_lut_np = lut_data[:, 1:4]
    scanner_lut = torch.from_numpy(scanner_lut_np).float()
    
    # Create an output directory for listmode data if it doesn't exist.
    lmf_output_dir = f'/mnt/d/fyq/sinogram/reconstruction_npy_full_train/{num_events:d}/listmode'
    output_dir = f'/mnt/d/fyq/sinogram/reconstruction_npy_full_train/{num_events:d}'
    os.makedirs(lmf_output_dir, exist_ok=True)
    
    # Create a log directory using the current timestamp
    log_dir = os.path.join("log", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=False)
    print(f"Log directory: {log_dir}")
    
    output_dir_sinogram = f'/mnt/d/fyq/sinogram/reconstruction_npy_full_train/{num_events:d}/sinogram'
    os.makedirs(output_dir_sinogram, exist_ok=True)

    
    # Create a thread pool for controlling concurrency
    from concurrent.futures import ThreadPoolExecutor
    # Use a reasonable number of threads based on your system (e.g., half the cores)
    max_parallel_saves = min(16, os.cpu_count())
    print(f"Using {max_parallel_saves} parallel file saving threads")
    
    # Keep track of all save threads to ensure they complete
    save_threads = []

    # Process each image file from 3d_image_0.npy to 3d_image_169.npy
    for i in range(45, 170):
        start_time_total = time.time()
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
        
        # Start saving events in background without blocking
        minimal_file = os.path.join(lmf_output_dir, f"listmode_data_minimal_{i}_{num_events}")
        save_thread = save_events_background(minimal_file, events, save_full_data=False)
        save_threads.append(save_thread)
        
        # Extract the index and num for the output filename
        out_name = f"reconstructed_index{i}_num{num_events}"
        
        # Convert raw events to detector IDs for reconstruction
        detector_ids_np = events  # Assumes events are already in correct format
        
        # Reconstruct directly from events, bypassing file load
        start_time = time.time()
        print("Starting reconstruction...")
        result_3d = reconstruct_volume_for_lmf(
            detector_ids_np=detector_ids_np,
            scanner_lut=scanner_lut,  # Pass directly, avoid reloading
            voxel_size=voxel_size,
            volume_size=volume_size,
            extended_size=extended_size,
            n_iters=n_iters,
            n_subsets=n_subsets,
            psf_fwhm_mm=psf_fwhm_mm,
            detector_outlier=outlier
        )
        
        # Save reconstruction result
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, result_3d)
        end_time = time.time()
        print(f"  -> Saved {out_path} (shape={result_3d.shape}) in {end_time - start_time:.1f}s")
        
        # Process sinogram directly from events (no need to reload)
        detector_ids = torch.from_numpy(events).to(torch.int32)
        sinogram_randoms_true = gate.listmode_to_sinogram(detector_ids, info)
        # del detector_ids
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
        # del sinogram_randoms_true
        
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

        end_time_total = time.time()
        print(f"Total time: {end_time_total - start_time_total}s")
        
        
        # Limit the number of concurrent save threads
        # Wait for some threads to complete if we've reached the limit
        while len([t for t in save_threads if t.is_alive()]) >= max_parallel_saves:
            time.sleep(0.5)  # Check every half second


    # Wait for all save threads to complete before exiting
    print("Waiting for all file save operations to complete...")
    for thread in save_threads:
        thread.join()

    print("\nAll reconstructions complete.")
    
if __name__ == "__main__":
    main()