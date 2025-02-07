#!/usr/bin/env python3
"""
reconstruction_all.py

This script recursively searches for minimal list-mode data (.lmf files) under
'listmode_test', e.g.:

  listmode_test/400000000/listmode_data_full_15_400000000.lmf

It then:
  1) Loads each file,
  2) Reads the detector LUT from 'detector_lut.txt',
  3) Reconstructs using OSEM,
  4) Applies iterative outlier removal & normalization,
  5) Saves each volume as a .npy file.

Usage:
  python reconstruction_all.py
"""

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

from outlier_detection import (
    global_outlier_detection,
    local_outlier_detection,
    edge_outlier_detection,
    combined_outlier_detection,
    analyze_outlier_masks,
    remove_outliers_iteratively
)

def reconstruct_volume_for_lmf(lmf_file: str,
                               lut_file: str,
                               voxel_size: float = 2.78,
                               volume_size: int = 128,
                               extended_size: int = 128,
                               n_iters: int = 2,
                               n_subsets: int = 34,
                               psf_fwhm_mm: float = 4.5,
                               detector_outlier: bool=True) -> np.ndarray:
    """
    Reconstruct a volume from a single minimal listmode (.lmf) file using OSEM,
    then apply outlier detection and iterative removal.

    Args:
        lmf_file (str): Path to the minimal .lmf file (det1_id, det2_id).
        lut_file (str): Path to the detector LUT (det_id, x, y, z).
        voxel_size (float): The voxel size in mm.
        volume_size (int): The final cropped volume dimension (cube).
        extended_size (int): The internal recon shape if you want to reconstruct
                             a larger or padded volume.
        n_iters (int): Number of OSEM iterations (each w/ n_subsets).
        n_subsets (int): Number of subsets in OSEM.
        psf_fwhm_mm (float): If >0, apply a Gaussian PSF with given FWHM in mm.

    Returns:
        patched_image (np.ndarray): 3D array [volume_size, volume_size, volume_size]
                                    after outlier removal and normalization.
    """

    # -------------------------------------------------------------------------
    # 1. Read the minimal listmode events from the .lmf file.
    # -------------------------------------------------------------------------
    dtype_minimal = np.dtype([('det1_id', np.int16), ('det2_id', np.int16)])
    events_np = np.fromfile(lmf_file, dtype=dtype_minimal)
    # Convert to a 2D array of shape (N, 2)
    detector_ids_np = np.column_stack((events_np['det1_id'], events_np['det2_id']))
    # Convert to torch.Tensor
    detector_ids = torch.from_numpy(detector_ids_np).long()

    # -------------------------------------------------------------------------
    # 2. Read the scanner LUT from the text file (skip 1 header line).
    # -------------------------------------------------------------------------
    lut_data = np.loadtxt(lut_file, skiprows=1)
    # Columns are [detector_id, x, y, z], but we only need x,y,z.
    scanner_lut_np = lut_data[:, 1:4]
    scanner_lut = torch.from_numpy(scanner_lut_np).float()

    # -------------------------------------------------------------------------
    # 3. Create PET listmode projection metadata.
    # -------------------------------------------------------------------------
    proj_meta = PETLMProjMeta(
        detector_ids=detector_ids,
        info=None,                   # Not provided.
        scanner_LUT=scanner_lut,     # Provide the LUT read from file.
        tof_meta=None,
        weights=None,
        detector_ids_sensitivity=None,
        weights_sensitivity=None
    )

    # -------------------------------------------------------------------------
    # 4. Define the reconstruction volume (ObjectMeta).
    # -------------------------------------------------------------------------
    object_meta = ObjectMeta(
        dr=(voxel_size, voxel_size, voxel_size),
        shape=(extended_size, extended_size, extended_size)
    )

    # -------------------------------------------------------------------------
    # 5. Create PET system matrix (optionally with PSF).
    # -------------------------------------------------------------------------
    psf_transform = GaussianFilter(psf_fwhm_mm)
    system_matrix = PETLMSystemMatrix(
        object_meta,
        proj_meta,
        obj2obj_transforms=[psf_transform],
        N_splits=8
    )

    # -------------------------------------------------------------------------
    # 6. Define the Poisson log-likelihood and OSEM algorithm.
    # -------------------------------------------------------------------------
    likelihood = PoissonLogLikelihood(system_matrix)
    recon_algorithm = OSEM(likelihood)

    # -------------------------------------------------------------------------
    # 7. Reconstruct the image using OSEM.
    # -------------------------------------------------------------------------
    # e.g., run `n_iters` OSEM iterations, each with `n_subsets` subsets.
    recon_image = recon_algorithm(n_iters=n_iters, n_subsets=n_subsets)

    # If extended_size > volume_size, crop around the center.
    diff = extended_size - volume_size
    start_idx = diff // 2
    end_idx = extended_size - (diff // 2)
    recon_image = recon_image[
        start_idx:end_idx,
        start_idx:end_idx,
        start_idx:end_idx
    ]

    
    # -------------------------------------------------------------------------
    # 8. Reorient the volume if needed.
    # We do a `permute` so that [z,y,x] -> [x,y,z], etc. 
    # But be sure to adapt to your needs. 
    # If you're consistent with indexing, you might skip this step.
    # -------------------------------------------------------------------------
    recon_image_align = recon_image.permute(2, 1, 0)
    
    # Convert to NumPy for outlier detection.
    recon_np = recon_image_align.cpu().numpy()
    
    # -------------------------------------------------------------------------
    # 9. Scale the reconstruction to [0,1] range.
    # -------------------------------------------------------------------------
    min_val = recon_np.min()
    max_val = recon_np.max()
    if max_val - min_val > 1e-9:
        recon_np_scaled = (recon_np - min_val) / (max_val - min_val)
    else:
        recon_np_scaled = recon_np.copy()

    # -------------------------------------------------------------------------
    # 10. Remove Outliers Iteratively (global + local + optional edge).
    # -------------------------------------------------------------------------
    # After the iterative removal, we get a "cleaned" image with outliers replaced.
    if detector_outlier:
        cleaned_image = remove_outliers_iteratively(
            recon_np_scaled,
            max_iters=10,
            global_factor=1.8, 
            global_percentile=99.0,
            local_window=3,
            local_sigma=3.0
        )
    else:
        cleaned_image = recon_np_scaled

    # Final scale to [0,1].
    cleaned_min = cleaned_image.min()
    cleaned_max = cleaned_image.max()
    if cleaned_max - cleaned_min > 1e-9:
        patched_image = (cleaned_image - cleaned_min) / (cleaned_max - cleaned_min)
    else:
        patched_image = cleaned_image.copy()

    # Return the final 3D array (shape [volume_size, volume_size, volume_size]).
    return patched_image


def main():
    """
    Main script to process all minimal list-mode data in a folder, reconstruct 
    each volume, apply outlier removal, and save the result to .npy files.
    
    """
    import glob
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmf_root", type=str, default="listmode_test",
                        help="Root folder containing subfolders with .lmf files.")
    parser.add_argument("--lut_file", type=str, default="detector_lut.txt",
                        help="Path to the detector LUT file.")
    parser.add_argument("--output_dir", type=str, default="reconstruction_npy_full",
                        help="Folder to save reconstructed volumes.")
    parser.add_argument("--outlier", type=bool, default=True)
    args = parser.parse_args()

    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Recursively find all .lmf files
    pattern = os.path.join(args.lmf_root, "*minimal*.lmf")
    lmf_files = sorted(glob.glob(pattern, recursive=False))

    print(f"Found {len(lmf_files)} .lmf files under {args.lmf_root}.")
    if not lmf_files:
        return

    # Regex to parse filename of the form: listmode_data_full_{index}_{num}.lmf
    # e.g.: listmode_data_full_15_400000000.lmf
    filename_regex = re.compile(r"listmode_data_minimal_(\d+)_(\d+)\.lmf")

    # Reconstruction parameters
    voxel_size = 2.78
    volume_size = 128
    extended_size = 128
    n_iters = 1
    n_subsets = 34
    psf_fwhm_mm = 4.5
    
    for i, lmf_path in enumerate(lmf_files):
        print(f"\n[{i+1}/{len(lmf_files)}] Processing {lmf_path}")
        start_time = time.time()

        base_name = os.path.basename(lmf_path)
        match = filename_regex.match(base_name)
        if match:
            index_str, num_str = match.groups()
            out_name = f"reconstructed_index{index_str}_num{num_str}.npy"
        else:
            # fallback if not matching
            out_name = "reconstructed_" + os.path.splitext(base_name)[0] + ".npy"

        # Reconstruct
        result_3d = reconstruct_volume_for_lmf(
            lmf_file=lmf_path,
            lut_file=args.lut_file,
            voxel_size=voxel_size,
            volume_size=volume_size,
            extended_size=extended_size,
            n_iters=n_iters,
            n_subsets=n_subsets,
            psf_fwhm_mm=psf_fwhm_mm,
            detector_outlier=args.outlier
        )

        # Save
        out_path = os.path.join(args.output_dir, out_name)
        np.save(out_path, result_3d)
        end_time = time.time()

        print(f"  -> Saved {out_path} (shape={result_3d.shape}) in {end_time - start_time:.1f}s")

    print("\nAll reconstructions complete.")



if __name__ == "__main__":
    main()


# python reconstruction_all.py --lmf_root listmode_test/400000000 --lut_file detector_lut.txt --output_dir reconstruction_npy_full_test/400000000