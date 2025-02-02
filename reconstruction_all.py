# main.py
import os
import time
import numpy as np
import torch
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter

def reconstruct_and_save(listmode_file, lut_file, output_file, n_iters=2, n_subsets=34):
    """
    For a given minimal listmode file, perform the PET reconstruction and save
    the reconstructed volume (aligned) as a .npy file.
    """
    # -----------------------------------------------------------------------------
    # 1. Read the minimal listmode events from the .lmf file.
    # -----------------------------------------------------------------------------
    # The minimal file contains only detector ID pairs (int16).
    dtype_minimal = np.dtype([('det1_id', np.int16), ('det2_id', np.int16)])
    events_np = np.fromfile(listmode_file, dtype=dtype_minimal)
    # Convert to a 2D array of shape (N, 2)
    detector_ids_np = np.column_stack((events_np['det1_id'], events_np['det2_id']))
    # Convert to a torch.Tensor.
    detector_ids = torch.from_numpy(detector_ids_np).long()

    # -----------------------------------------------------------------------------
    # 2. Read the scanner lookup table (LUT) from the text file.
    # -----------------------------------------------------------------------------
    # The LUT file is assumed to have a header, then rows: "detector_id x y z".
    lut_data = np.loadtxt(lut_file, skiprows=1)
    # The LUT has the detector id in column 0 and x,y,z in columns 1-3.
    scanner_lut_np = lut_data[:, 1:4]
    scanner_lut = torch.from_numpy(scanner_lut_np).float()

    # -----------------------------------------------------------------------------
    # 3. Create the PET listmode projection metadata.
    # -----------------------------------------------------------------------------
    proj_meta = PETLMProjMeta(
        detector_ids=detector_ids,
        info=None,                   # Not provided.
        scanner_LUT=scanner_lut,     # Provide the LUT read from file.
        tof_meta=None,
        weights=None,
        detector_ids_sensitivity=None,
        weights_sensitivity=None
    )

    # -----------------------------------------------------------------------------
    # 4. Define the object space (reconstruction volume).
    # -----------------------------------------------------------------------------
    # We want to reconstruct a 128x128x128 volume with voxel size 2.78 mm.
    object_meta = ObjectMeta(
        dr=(2.78, 2.78, 2.78),
        shape=(128, 128, 128)
    )

    # -----------------------------------------------------------------------------
    # 5. Create the PET system matrix.
    # -----------------------------------------------------------------------------
    # Optionally, apply a Gaussian PSF with 4.5 mm FWHM.
    psf_transform = GaussianFilter(4.5)
    system_matrix = PETLMSystemMatrix(
        object_meta,
        proj_meta,
        obj2obj_transforms=[psf_transform],
        N_splits=8
    )

    # -----------------------------------------------------------------------------
    # 6. Define the Poisson log-likelihood.
    # -----------------------------------------------------------------------------
    likelihood = PoissonLogLikelihood(system_matrix)

    # -----------------------------------------------------------------------------
    # 7. Reconstruct the image using the OSEM algorithm.
    # -----------------------------------------------------------------------------
    recon_algorithm = OSEM(likelihood)
    recon_image = recon_algorithm(n_iters=n_iters, n_subsets=n_subsets)
    
    # -----------------------------------------------------------------------------
    # 8. Align the reconstructed image.
    # -----------------------------------------------------------------------------
    # Based on your requirements, we want the index ordering to match that of the original:
    #    orig_image[:, :, index]  <==>  recon_image_align[:, :, index]
    # Assuming that the reconstructed image's axes need to be permuted, we do:
    recon_image_align = recon_image.permute(2, 1, 0)
    
    # -----------------------------------------------------------------------------
    # 9. Save the reconstructed volume as a .npy file.
    # -----------------------------------------------------------------------------
    np.save(output_file, recon_image_align.cpu().numpy())

def main():
    # Folder containing the 170 minimal listmode data files.
    listmode_folder = r"listmode_test"
    # File for the scanner lookup table (LUT) (assumed to be in the current directory)
    lut_file = "detector_lut.txt"
    # Output folder where the reconstructed .npy files will be saved.
    output_folder = "reconstruction_npy_full_test"
    os.makedirs(output_folder, exist_ok=True)
    
    num_files = 170  # From 0 to 169.
    for i in range(num_files):
        listmode_file = os.path.join(listmode_folder, f"listmode_data_minimal_{i}_100000000.lmf")
        output_file = os.path.join(output_folder, f"reconstructed_{i}_128x128x128.npy")
        print(f"Processing file: {listmode_file}")
        start_time = time.time()
        reconstruct_and_save(listmode_file, lut_file, output_file)
        end_time = time.time()
        print(f"Processed file {i} in {end_time - start_time:.2f} seconds. Saved to {output_file}")

if __name__ == "__main__":
    main()
