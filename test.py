from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Use a more complicated image (Shepp-Logan Phantom)
def generate_phantom_image(size=128):
    """
    Generate a more complex image for testing (Shepp-Logan Phantom).
    """
    phantom = shepp_logan_phantom()
    phantom_resized = resize(phantom, (size, size), mode='reflect', anti_aliasing=True)
    return phantom_resized

# Step 2: Simulate a full ring of detectors using Radon transform
def simulate_detectors(image):
    """
    Simulate a full ring of detectors using Radon transform.
    """
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    return sinogram, theta

# Step 3: Recover the image using inverse Radon transform
def recover_image(sinogram, theta):
    """
    Recover the image from the Radon transform using inverse Radon transform.
    """
    reconstructed_image = iradon(sinogram, theta=theta, circle=True)
    return reconstructed_image

# Generate a more complex image
complex_image = generate_phantom_image()

# Simulate a full ring of detectors
sinogram, theta = simulate_detectors(complex_image)

# Recover the image from the detector data
recovered_image = recover_image(sinogram, theta)

# Plot the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Complex Image")
plt.imshow(complex_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Sinogram (Detector Data)")
plt.imshow(sinogram, cmap='gray', aspect='auto')
plt.xlabel("Angles (Theta)")
plt.ylabel("Detector")

plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
plt.imshow(recovered_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
