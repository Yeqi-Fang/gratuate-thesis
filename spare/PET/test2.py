import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq

def create_detector_ring(num_detectors, radius):
    angles = np.linspace(0, 2*np.pi, num_detectors, endpoint=False)
    detector_positions = np.vstack((radius * np.cos(angles), radius * np.sin(angles))).T
    return detector_positions, angles

def create_radioactive_sources(image_size, num_sources=5):
    image = np.zeros((image_size, image_size))
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    X, Y = np.meshgrid(x, y)
    
    for _ in range(num_sources):
        shape = np.random.choice(['circle', 'ellipse'])
        if shape == 'circle':
            # Random center and radius
            cx, cy = np.random.uniform(-0.8, 0.8, 2)
            r = np.random.uniform(0.05, 0.15)
            mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        else:
            # Random center, axes, and rotation
            cx, cy = np.random.uniform(-0.8, 0.8, 2)
            a, b = np.random.uniform(0.05, 0.15, 2)
            theta = np.random.uniform(0, np.pi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            mask = (((X - cx) * cos_theta + (Y - cy) * sin_theta)**2) / a**2 + \
                   (((X - cx) * sin_theta - (Y - cy) * cos_theta)**2) / b**2 <= 1
        intensity = np.random.uniform(0.5, 1.0)
        image[mask] += intensity
    # Normalize the image
    image = image / np.max(image)
    return image

def simulate_emissions(image, num_events, detector_positions, angles, detector_radius):
    image_size = image.shape[0]
    sinogram = np.zeros((len(angles), len(angles)))
    
    # Define the activity distribution
    activity = image / np.sum(image)
    
    # Flatten the activity for selection
    activity_flat = activity.flatten()
    pixels = np.arange(activity.size)
    
    for _ in range(num_events):
        # Select a pixel based on activity
        pixel = np.random.choice(pixels, p=activity_flat)
        ix, iy = np.unravel_index(pixel, image.shape)
        
        # Convert pixel index to coordinates
        x = (ix - image_size / 2) / (image_size / 2)
        y = (iy - image_size / 2) / (image_size / 2)
        
        # Random emission angle
        phi = np.random.uniform(0, 2*np.pi)
        # Photon 1 direction
        dir1 = np.array([np.cos(phi), np.sin(phi)])
        # Photon 2 direction (opposite)
        dir2 = -dir1
        
        # Detect which detectors detect the photons
        # Compute angles
        angle1 = np.arctan2(dir1[1], dir1[0]) % (2*np.pi)
        angle2 = np.arctan2(dir2[1], dir2[0]) % (2*np.pi)
        
        # Find the closest detectors
        idx1 = np.argmin(np.abs(angles - angle1))
        idx2 = np.argmin(np.abs(angles - angle2))
        
        # Increment the sinogram
        sinogram[idx1, idx2] += 1
        sinogram[idx2, idx1] += 1  # Since the pair is bidirectional
    
    return sinogram


def filtered_backprojection(sinogram, num_detectors, image_size):
    # Initialize the reconstructed image
    reconstruction = np.zeros((image_size, image_size))
    
    # Define the angles
    angles = np.linspace(0., 180., num_detectors, endpoint=False)
    
    # Define the filter (Ram-Lak)
    n = max(64, int(2 ** np.ceil(np.log2(2 * sinogram.shape[0]))))
    freq = fftfreq(n).reshape(-1, 1)
    filter = 2 * np.abs(freq)
    
    # Apply the filter to each projection
    filtered_sinogram = np.zeros_like(sinogram)
    for i in range(sinogram.shape[1]):
        projection = sinogram[:, i]
        projection_padded = np.pad(projection, (0, n - len(projection)), mode='constant', constant_values=0)
        projection_fft = fft(projection_padded)
        projection_filtered = np.real(ifft(projection_fft * filter[:,0]))
        filtered = projection_filtered[:len(projection)]
        filtered_sinogram[:, i] = filtered
    
    # Backprojection
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    X, Y = np.meshgrid(x, y)
    
    for i, angle in enumerate(angles):
        theta = np.deg2rad(angle)
        t = X * np.cos(theta) + Y * np.sin(theta)
        # Interpolate the filtered projection
        interp = interp1d(np.linspace(-1, 1, len(filtered_sinogram[:,i])), 
                          filtered_sinogram[:,i], bounds_error=False, fill_value=0)
        projection = interp(t)
        reconstruction += projection
    
    # Normalize the image
    reconstruction = reconstruction / np.max(reconstruction)
    return reconstruction


def pet_simulation():
    # Parameters
    num_detectors = 180  # Number of detectors in the ring
    detector_radius = 1.0  # Radius of the detector ring
    image_size = 128  # Size of the reconstructed image
    num_sources = 5  # Number of radioactive sources
    num_events = 100000  # Number of annihilation events to simulate
    
    # Step 1: Create detector ring
    detector_positions, angles = create_detector_ring(num_detectors, detector_radius)
    
    # Step 2: Create radioactive sources
    image = create_radioactive_sources(image_size, num_sources)
    
    # Display the radioactive sources
    plt.figure(figsize=(6,6))
    plt.imshow(image, extent=[-1,1,-1,1], cmap='hot')
    plt.title('Radioactive Sources')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Activity')
    plt.scatter(detector_positions[:,0], detector_positions[:,1], c='blue', s=10, label='Detectors')
    plt.legend()
    plt.show()
    
    # Step 3: Simulate emissions and detection
    sinogram = simulate_emissions(image, num_events, detector_positions, angles, detector_radius)
    
    # Display the sinogram
    plt.figure(figsize=(8,6))
    plt.imshow(sinogram, aspect='auto', cmap='gray')
    plt.title('Sinogram')
    plt.xlabel('Detector Pair Index')
    plt.ylabel('Detector Index')
    plt.colorbar(label='Counts')
    plt.show()
    
    # Step 4: Reconstruct the image
    reconstructed_image = filtered_backprojection(sinogram, num_detectors, image_size)
    
    # Display the reconstructed image
    plt.figure(figsize=(6,6))
    plt.imshow(reconstructed_image, extent=[-1,1,-1,1], cmap='hot')
    plt.title('Reconstructed Image')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Normalized Activity')
    plt.show()
    
if __name__ == "__main__":
    pet_simulation()
