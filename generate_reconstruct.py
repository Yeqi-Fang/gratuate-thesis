# generate_reconstruct.py

import os
import re
import glob
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter

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
    'radius': np.float32(290.56),
    'crystalTransNr': 13,
    'crystalTransSpacing': np.float32(4.03125),
    'crystalAxialNr': 7,
    'crystalAxialSpacing': np.float32(5.36556),
    'submoduleAxialNr': 1,
    'submoduleAxialSpacing': np.float32(0.0),
    'submoduleTransNr': 1,
    'submoduleTransSpacing': np.float32(0.0),
    'moduleTransNr': 1,
    'moduleTransSpacing': np.float32(0.0),
    'moduleAxialNr': 6,
    'moduleAxialSpacing': np.float32(89.82),
    'rsectorTransNr': 32,
    'rsectorAxialNr': 1,
    'TOF': 1,
    'num_tof_bins': np.float32(29.0),
    'tof_range': np.float32(735.7705),
    'tof_fwhm': np.float32(57.71),
    'NrCrystalsPerRing': 416,
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
num_events = int(5e9)
save_events_pos = False

# Create PET scanner geometry from info
geometry = create_pet_geometry(info)

# Save the detector lookup table once.
# (Since LUT depends only on geometry, a dummy image is sufficient.)
dummy_image = np.ones((1,1,1), dtype=np.float32)
simulator_for_lut = PETSimulator(geometry, dummy_image, voxel_size=2.78)
simulator_for_lut.save_detector_positions("detector_lut.txt")

# Define the base directory where the 3D image files are stored.
base_dir = r"D:\Datasets\dataset\test_npy_crop"
lut_file = 'detector_lut.txt'
# Create an output directory for listmode data if it doesn't exist.
lmf_output_dir = "tmp"
output_dir = f'reconstruction_npy_full_train/{num_events:d}'
os.makedirs(lmf_output_dir, exist_ok=True)

# Process each image file from 3d_image_0.npy to 3d_image_169.npy
for i in range(36):
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

    filename_regex = re.compile(r"listmode_data_minimal_(\d+)_(\d+)\.npz")
    print(f"\n[{i+1}/{len(lmf_output_dir)}] Processing {lmf_output_dir}")
    start_time = time.time()
    
    lmf_path = minimal_file + ".npz"
    
    # base_name = os.path.basename(lmf_output_dir)
    match = filename_regex.match(lmf_path)
    if match:
        index_str, num_str = match.groups()
        out_name = f"reconstructed_index{index_str}_num{num_str}.npz"
    else:
        # fallback if not matching
        out_name = "reconstructed_" + os.path.splitext(lmf_path)[0] + ".npz"

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
    out_path = os.path.join(output_dir, out_name)
    np.save(out_path, result_3d)
    
    os.remove(lmf_path)
    end_time = time.time()

    print(f"  -> Saved {out_path} (shape={result_3d.shape}) in {end_time - start_time:.1f}s")

print("\nAll reconstructions complete.")