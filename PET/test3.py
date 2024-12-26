import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

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
        main_circle = x*x + y*y <= (self.image_size//3)**2
        self.phantom[main_circle] = 0.3
        
        # Add random hot spots (more radioactive regions)
        for _ in range(3):
            center_x = random.randint(self.image_size//4, 3*self.image_size//4)
            center_y = random.randint(self.image_size//4, 3*self.image_size//4)
            radius = random.randint(10, 30)
            
            y, x = np.ogrid[-center_y:self.image_size-center_y, 
                           -center_x:self.image_size-center_x]
            hot_spot = x*x + y*y <= radius**2
            self.phantom[hot_spot] = random.uniform(0.6, 1.0)
        
        # Smooth the phantom
        self.phantom = gaussian_filter(self.phantom, sigma=2)

    def _simulate_batch(self, n_events, seed):
        """Simulate a batch of events for parallel processing"""
        np.random.seed(seed)
        random.seed(seed)
        local_sinogram = np.zeros((self.n_detectors // 2, self.n_detectors))
        
        probs = self.phantom.flatten() / self.phantom.sum()
        
        for _ in range(n_events):
            # Randomly select emission point based on activity
            emission_idx = np.random.choice(self.image_size**2, p=probs)
            emission_y, emission_x = np.unravel_index(emission_idx, self.phantom.shape)
            
            # Random angle for emission
            angle = random.uniform(0, 2*np.pi)
            
            # Find closest detectors for both photons
            direction = np.array([np.cos(angle), np.sin(angle)])
            pos1 = np.array([emission_x, emission_y])
            pos2 = pos1 + direction * self.image_size
            
            # Find nearest detectors
            d1 = np.argmin(np.sum((self.detector_positions - pos1)**2, axis=1))
            d2 = np.argmin(np.sum((self.detector_positions - pos2)**2, axis=1))
            
            # Record detection in sinogram
            if abs(d1 - d2) > self.n_detectors//2:
                continue
            row = min(d1, d2)
            col = max(d1, d2)
            local_sinogram[row % (self.n_detectors//2), col] += 1
            
        return local_sinogram

    def simulate_detection(self, n_events=1000000):
        """Simulate photon emission and detection using multiple processes"""
        n_cores = multiprocessing.cpu_count()
        events_per_process = n_events // n_cores
        
        # Create a partial function with fixed parameters
        simulate_func = partial(self._simulate_batch, events_per_process)
        
        # Generate different seeds for each process
        seeds = [random.randint(0, 10000) for _ in range(n_cores)]
        
        # Run simulation in parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(simulate_func, seeds))
        
        # Combine results
        self.sinogram = sum(results)

    def _process_reconstruction_batch(self, detector_pairs):
        """Process a batch of detector pairs for reconstruction"""
        partial_reconstruction = np.zeros((self.image_size, self.image_size))
        y_grid, x_grid = np.mgrid[0:self.image_size, 0:self.image_size]
        
        for i, j in detector_pairs:
            if self.sinogram[i, j] == 0:
                continue
                
            # Get detector positions
            d1 = self.detector_positions[i]
            d2 = self.detector_positions[j]
            
            # Create line between detectors
            direction = d2 - d1
            length = np.sqrt(np.sum(direction**2))
            direction = direction / length
            
            # Calculate perpendicular distance from each pixel to the line
            v = np.column_stack((x_grid.flatten(), y_grid.flatten())) - d1
            dist = np.abs(np.cross(v, direction)) / length
            
            # Add contribution to pixels near the line
            contribution = np.exp(-dist**2 / 2) * self.sinogram[i, j]
            partial_reconstruction += contribution.reshape(self.image_size, self.image_size)
            
        return partial_reconstruction

    def reconstruct_image(self):
        """Parallel implementation of backprojection reconstruction"""
        # Create list of all detector pairs
        detector_pairs = [(i, j) for i in range(self.n_detectors // 2) 
                         for j in range(self.n_detectors)]
        
        # Split pairs into batches for parallel processing
        n_cores = multiprocessing.cpu_count()
        batch_size = len(detector_pairs) // n_cores
        batches = [detector_pairs[i:i + batch_size] 
                  for i in range(0, len(detector_pairs), batch_size)]
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(self._process_reconstruction_batch, batches))
        
        # Combine results
        reconstruction = sum(results)
        return gaussian_filter(reconstruction, sigma=1)

    def visualize(self, reconstruction):
        """Visualize original phantom, sinogram, and reconstruction"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original phantom
        axes[0].imshow(self.phantom, cmap='hot')
        axes[0].scatter(self.detector_positions[:, 0], 
                       self.detector_positions[:, 1], 
                       c='blue', s=10, alpha=0.5)
        axes[0].set_title('Original Phantom')
        
        # Plot sinogram
        axes[1].imshow(np.log1p(self.sinogram), cmap='hot')
        axes[1].set_title('Sinogram (log scale)')
        
        # Plot reconstruction
        axes[2].imshow(reconstruction, cmap='hot')
        axes[2].set_title('Reconstructed Image')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Run simulation
    simulator = PETSimulator(n_detectors=360, image_size=200)
    simulator.create_phantom()
    
    print("Starting simulation...")
    simulator.simulate_detection(n_events=1000000)
    print("Simulation complete. Starting reconstruction...")
    
    reconstruction = simulator.reconstruct_image()
    print("Reconstruction complete. Visualizing results...")
    
    simulator.visualize(reconstruction)