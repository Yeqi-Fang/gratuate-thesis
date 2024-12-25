# generate_dataset.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk, ellipse
from skimage.transform import radon
import concurrent.futures


def generate_random_phantom(size=128, max_shapes=5):
    """Create a random phantom image with horizontally aligned shapes."""
    image = np.zeros((size, size), dtype=np.float32)
    num_shapes = np.random.randint(1, max_shapes + 1)
    for _ in range(num_shapes):
        shape_type = np.random.choice(["circle", "ellipse"])
        center_x = np.random.randint(size // 4, 3 * size // 4)
        center_y = np.random.randint(0, size)
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


def create_incomplete_sinogram(image, angles=None, missing_angle_start=30, missing_angle_end=60):
    """
    Compute Radon transform and zero out sinogram columns corresponding to angles
    between missing_angle_start and missing_angle_end degrees to create vertical missing regions.
    """
    if angles is None:
        angles = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=angles, circle=True)
    # print(angles.shape)
    # Zero out columns where angles are in [missing_angle_start, missing_angle_end)
    mask = (angles >= missing_angle_start) & (angles < missing_angle_end)
    incomplete = sinogram.copy()
    incomplete[:, mask] = 0.0  # Zero out columns to create vertical missing regions
    # print(sinogram.shape)
    return sinogram, incomplete


def _process_single_sample(
    i,
    size,
    angles,
    missing_angle_start,
    missing_angle_end,
    out_folder
):
    """
    Generates a single sample (phantom, sinograms), saves .npy files and debug .png.
    This function is meant to be run in a separate process.
    """
    phantom = generate_random_phantom(size=size)
    complete_sino, incomplete_sino = create_incomplete_sinogram(
        phantom,
        angles=angles,
        missing_angle_start=missing_angle_start,
        missing_angle_end=missing_angle_end
    )
    
    # Save the arrays as .npy
    incomplete_path = os.path.join(out_folder, f"incomplete_{i}.npy")
    complete_path = os.path.join(out_folder, f"complete_{i}.npy")
    np.save(incomplete_path, incomplete_sino)
    np.save(complete_path, complete_sino)

    # Optional: Save a debug .png
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(phantom, cmap='gray', extent=[0, 180, 0, 128])
    axs[0].set_title("Phantom")
    # axs[0].axis('off')

    axs[1].imshow(incomplete_sino, cmap='gray', extent=[0, 180, 0, 128])
    axs[1].set_title("Incomplete Sinogram")
    # axs[1].axis('off')

    axs[2].imshow(complete_sino, cmap='gray', extent=[0, 180, 0, 128])
    axs[2].set_title("Complete Sinogram")
    # axs[2].axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_folder, f"sample_{i}.png")
    plt.savefig(out_path)
    plt.close(fig)


def generate_dataset(
    num_samples,
    size,
    missing_angle_start,
    missing_angle_end,
    out_folder,
    use_multiprocessing=False
):
    """
    Generate a dataset of incomplete/complete sinograms, saving them as .npy and .png files.
    
    Parameters:
    - num_samples (int): Number of samples to generate.
    - size (int): Phantom image size (e.g., 128).
    - missing_angle_start (int): Start angle for missing region (degrees).
    - missing_angle_end (int): End angle for missing region (degrees).
    - out_folder (str): Directory to save generated data.
    - use_multiprocessing (bool): If True, use multiprocessing to generate data.
    """
    os.makedirs(out_folder, exist_ok=True)
    angles = np.linspace(0., 180., size, endpoint=False)

    if use_multiprocessing:
        print("Generating dataset using multiprocessing...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_samples):
                futures.append(executor.submit(
                    _process_single_sample,
                    i,
                    size,
                    angles,
                    missing_angle_start,
                    missing_angle_end,
                    out_folder
                ))
            # Optionally, show progress
            for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    fut.result()
                    if idx % 10 == 0 or idx == num_samples:
                        print(f"Generated {idx}/{num_samples} samples...")
                except Exception as e:
                    print(f"Sample {idx} generated with exception: {e}")
    else:
        print("Generating dataset sequentially...")
        for i in range(num_samples):
            _process_single_sample(
                i,
                size,
                angles,
                missing_angle_start,
                missing_angle_end,
                out_folder
            )
            if (i + 1) % 10 == 0 or (i + 1) == num_samples:
                print(f"Generated {i + 1}/{num_samples} samples...")


def main():
    parser = argparse.ArgumentParser(description="Generate sinogram dataset and save to a folder.")
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate.')
    parser.add_argument('--size', type=int, default=128, help='Phantom image size.')
    parser.add_argument('--missing_angle_start', type=int, default=30,
                        help='Start angle for missing region (degrees).')
    parser.add_argument('--missing_angle_end', type=int, default=60,
                        help='End angle for missing region (degrees).')
    parser.add_argument('--out_folder', type=str, default='dataset_folder',
                        help='Folder to store the generated dataset.')
    parser.add_argument('--use_multiprocessing', action='store_true',
                        help='Enable multiprocessing for dataset generation.')
    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        size=args.size,
        missing_angle_start=args.missing_angle_start,
        missing_angle_end=args.missing_angle_end,
        out_folder=args.out_folder,
        use_multiprocessing=args.use_multiprocessing
    )
    print(f"Dataset generated and saved in {args.out_folder}")


if __name__ == "__main__":
    main()
