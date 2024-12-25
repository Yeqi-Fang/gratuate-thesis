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


def create_incomplete_sinogram(image, angles=None, missing_row_start=40, missing_row_end=60):
    """Compute radon transform and zero out a horizontal band."""
    if angles is None:
        angles = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=angles, circle=False)
    sinogram = sinogram.T
    incomplete = sinogram.copy()
    incomplete[missing_row_start:missing_row_end, :] = 0.0
    return sinogram, incomplete


def _process_single_sample(
    i,
    size,
    angles,
    missing_row_start,
    missing_row_end,
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
        missing_row_start=missing_row_start,
        missing_row_end=missing_row_end
    )
    # Save the arrays as .npy
    incomplete_path = os.path.join(out_folder, f"incomplete_{i}.npy")
    complete_path = os.path.join(out_folder, f"complete_{i}.npy")
    np.save(incomplete_path, incomplete_sino)
    np.save(complete_path, complete_sino)

    # Optional: Save a debug .png
    fig, axs = plt.subplots(1, 3, figsize=(9,3))
    axs[0].imshow(phantom, cmap='gray')
    axs[0].set_title("Phantom")
    axs[1].imshow(incomplete_sino, cmap='gray')
    axs[1].set_title("Incomplete")
    axs[2].imshow(complete_sino, cmap='gray')
    axs[2].set_title("Complete")
    for ax in axs:
        ax.axis('off')
    plt.savefig(os.path.join(out_folder, f"sample_{i}.png"))
    plt.close(fig)


def generate_dataset(num_samples, size, missing_row_start, missing_row_end, out_folder):
    """
    Generate a dataset of incomplete/complete sinograms in multiprocessing mode,
    saving them as .npy (and .png debug images).
    """
    os.makedirs(out_folder, exist_ok=True)
    angles = np.linspace(0., 180., size, endpoint=False)

    # Use ProcessPoolExecutor to parallelize sample generation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(num_samples):
            futures.append(executor.submit(
                _process_single_sample,
                i,
                size,
                angles,
                missing_row_start,
                missing_row_end,
                out_folder
            ))
        # Wait for all jobs to complete
        for fut in concurrent.futures.as_completed(futures):
            # No return value needed here, just ensuring completion
            fut.result()


def main():
    parser = argparse.ArgumentParser(description="Generate sinogram dataset and save to a folder (multiprocessing).")
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate.')
    parser.add_argument('--size', type=int, default=128, help='Phantom image size.')
    parser.add_argument('--missing_row_start', type=int, default=40)
    parser.add_argument('--missing_row_end', type=int, default=60)
    parser.add_argument('--out_folder', type=str, default='dataset_folder',
                        help='Folder to store the generated dataset.')
    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        size=args.size,
        missing_row_start=args.missing_row_start,
        missing_row_end=args.missing_row_end,
        out_folder=args.out_folder
    )
    print(f"Dataset generated and saved in {args.out_folder}")


if __name__ == "__main__":
    main()
