import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

class PETSimulator:
    def __init__(self, n_detectors=360, image_size=200):
        self.n_detectors = n_detectors
        self.image_size = image_size
        self.detector_positions = self._create_detector_ring()
        self.phantom = np.zeros((image_size, image_size))
        self.sinogram = np.zeros((n_detectors // 2, n_detectors))
        
    def _create_detector_ring(self):
        """Create a ring of detectors"""
        angles = np.linspace(0, 2*np.pi, self.n_detectors, endpoint=False)
        radius = self.image_size // 2 - 10
        x = radius * np.cos(angles) + self.image_size // 2
        y = radius * np.sin(angles) + self.image_size // 2
        return np.column_stack((x, y))

    def create_phantom(self):
        """Create a phantom with random radioactive sources"""
        # Create main circular outline
        y, x = np.ogrid[-self.image_size//2:self.image_size//2, 
                        -self.image_size//2:self.image_size//2]
        main_circle = x*x + y*y <= (self.image_size//3)**2  # Main boundary
        self.phantom[main_circle] = 0.3

        # Add random hot spots (smaller circles)
        np.random.seed(42)  # For reproducibility
        for _ in range(3):  # Generate 3 hot spots
            while True:
                # Random center within the boundary
                center_x = random.randint(self.image_size//3, 2*self.image_size//3)
                center_y = random.randint(self.image_size//3, 2*self.image_size//3)
                radius = random.randint(10, 20)  # Reduced radius for smaller circles

                # Create a mask for the smaller circle
                y_offset, x_offset = np.ogrid[-center_y:self.image_size-center_y, 
                                            -center_x:self.image_size-center_x]
                hot_spot = x_offset*x_offset + y_offset*y_offset <= radius**2

                # Calculate the distance of the smaller circle's center to the center of the main circle
                distance_to_center = np.sqrt((center_x - self.image_size//2)**2 + (center_y - self.image_size//2)**2)

                # Ensure the smaller circle lies completely inside the main circle boundary
                if distance_to_center + radius <= self.image_size//3:
                    break  # If the circle is inside the main boundary, break the loop

            # Add the hot spot to the phantom
            self.phantom[hot_spot] = random.uniform(0.7, 1.0)
        
        # Smooth the phantom
        self.phantom = gaussian_filter(self.phantom, sigma=1)

    def _compute_line_integral(self, start, end, image):
        """Compute line integral between two points using ray tracing"""
        # Generate points along the line
        length = int(np.ceil(np.sqrt(np.sum((end - start) ** 2))))
        t = np.linspace(0, 1, length * 10)  # Increased sampling density
        points = start[None, :] + t[:, None] * (end - start)[None, :]
        
        # Convert to pixel coordinates
        x = points[:, 0].astype(int)
        y = points[:, 1].astype(int)
        
        # Filter valid points
        mask = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        x = x[mask]
        y = y[mask]
        
        # Sum up values along the line
        if len(x) > 0:
            return np.sum(image[y, x])  # Changed from mean to sum
        return 0

    def _simulate_batch(self, args):
        """Simulate a batch of detector pairs for parallel processing"""
        detector_pairs, phantom = args
        local_sinogram = np.zeros((self.n_detectors // 2, self.n_detectors))
        
        for i, j in detector_pairs:
            # Get detector positions
            d1 = self.detector_positions[i]
            d2 = self.detector_positions[j]
            
            # Compute line integral
            integral = self._compute_line_integral(d1, d2, phantom)
            
            # Store in sinogram
            row = min(i, j) % (self.n_detectors // 2)
            col = max(i, j)
            local_sinogram[row, col] = integral
            
        return local_sinogram

    def simulate_detection(self):
        """Simulate PET detection using line integrals"""
        # Generate all possible detector pairs
        detector_pairs = [(i, j) for i in range(self.n_detectors) 
                        for j in range(i + 1, self.n_detectors)]
        
        # Split work for parallel processing
        n_cores = multiprocessing.cpu_count()
        chunk_size = len(detector_pairs) // n_cores
        batches = [detector_pairs[i:i + chunk_size] 
                for i in range(0, len(detector_pairs), chunk_size)]
        
        # Prepare arguments for parallel processing
        args = [(batch, self.phantom) for batch in batches]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(self._simulate_batch, args))
            
        # Combine results
        self.sinogram = sum(results)
        
        # Normalize sinogram: each projection is normalized separately
        for col in range(self.n_detectors):
            col_max = self.sinogram[:, col].max()
            if col_max > 0:
                self.sinogram[:, col] /= col_max

    def _backproject_batch(self, args):
        """Backproject a batch of sinogram rows"""
        rows, detector_positions, sinogram = args
        image_size = self.image_size
        partial_reconstruction = np.zeros((image_size, image_size))
        
        y_coords, x_coords = np.meshgrid(np.arange(image_size), 
                                       np.arange(image_size), 
                                       indexing='ij')
        pixel_positions = np.stack([x_coords, y_coords], axis=-1)
        
        for row in rows:
            for col in range(self.n_detectors):
                if sinogram[row, col] == 0:
                    continue
                
                # Get detector positions
                d1 = detector_positions[row]
                d2 = detector_positions[col]
                
                # Calculate line parameters
                direction = d2 - d1
                length = np.sqrt(np.sum(direction**2))
                if length < 1e-10:
                    continue
                direction = direction / length
                
                # Calculate perpendicular distances
                v = pixel_positions - d1
                dist = np.abs(np.cross(v, direction.reshape(1, 1, 2)))
                
                # Add weighted contribution
                weight = np.exp(-dist**2 / 2)
                partial_reconstruction += weight * sinogram[row, col]
        
        return partial_reconstruction

    def reconstruct_image(self):
        """Reconstruct image using filtered backprojection"""
        # Split work for parallel processing
        n_cores = multiprocessing.cpu_count()
        rows = np.arange(self.n_detectors // 2)
        chunks = np.array_split(rows, n_cores)
        
        # Prepare arguments
        args = [(chunk, self.detector_positions, self.sinogram) 
                for chunk in chunks]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(self._backproject_batch, args))
        
        # Combine results
        reconstruction = sum(results)
        
        # Post-process reconstruction
        if reconstruction.max() > 0:
            reconstruction = reconstruction / reconstruction.max()
        reconstruction = gaussian_filter(reconstruction, sigma=1)
        
        return reconstruction

    def visualize(self, reconstruction):
        """Visualize original phantom, sinogram, and reconstruction"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original phantom
        im1 = axes[0].imshow(self.phantom, cmap='hot')
        axes[0].scatter(self.detector_positions[:, 0], 
                       self.detector_positions[:, 1], 
                       c='blue', s=10, alpha=0.5)
        axes[0].set_title('Original Phantom')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot sinogram
        im2 = axes[1].imshow(np.log1p(self.sinogram), cmap='hot')
        axes[1].set_title('Sinogram (log scale)')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot reconstruction
        im3 = axes[2].imshow(reconstruction, cmap='hot')
        axes[2].set_title('Reconstructed Image')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Run simulation
    simulator = PETSimulator(n_detectors=360, image_size=200)
    print("Creating phantom...")
    simulator.create_phantom()
    
    print("Simulating detection...")
    simulator.simulate_detection()
    
    print("Reconstructing image...")
    reconstruction = simulator.reconstruct_image()
    
    print("Visualizing results...")
    simulator.visualize(reconstruction)