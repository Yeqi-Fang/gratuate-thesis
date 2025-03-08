import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import time

class PETSimulation:
    def __init__(self, detector_radius=100, n_detectors=180):
        self.detector_radius = detector_radius
        self.n_detectors = n_detectors
        
        # Create detector positions (equally spaced around the circle)
        self.detector_angles = np.linspace(0, 2*np.pi, n_detectors, endpoint=False)
        self.detector_x = detector_radius * np.cos(self.detector_angles)
        self.detector_y = detector_radius * np.sin(self.detector_angles)
        
        # Initialize sinogram (now n_detectors × n_angles)
        self.n_angles = 180  # Number of angular bins (0 to 180 degrees)
        self.sinogram = np.zeros((n_detectors, self.n_angles))
        
        # Create density function (phantom)
        self.image_size = 200
        self.pixel_size = 2 * detector_radius / self.image_size
        self.phantom = self.create_phantom()
        
    def create_phantom(self):
        # [Previous phantom creation code remains the same]
        x = np.linspace(-self.detector_radius, self.detector_radius, self.image_size)
        y = np.linspace(-self.detector_radius, self.detector_radius, self.image_size)
        X, Y = np.meshgrid(x, y)
        
        phantom = np.zeros((self.image_size, self.image_size))
        
        # Create main circle
        main_circle = (X**2 + Y**2) <= (0.8 * self.detector_radius)**2
        phantom[main_circle] = 1
        
        # Add hot spots
        hot_spot1 = ((X - 20)**2 + (Y - 20)**2) <= 15**2
        phantom[hot_spot1] = 7
        
        hot_spot2 = ((X + 30)**2 + (Y + 30)**2) <= 20**2
        phantom[hot_spot2] = 5
        
        ellipse = ((X - 10)**2/400 + (Y + 20)**2/100) <= 1
        phantom[ellipse] = 10
        
        return phantom

    @staticmethod
    def line_circle_intersection(x0, y0, theta, radius):
        # [Previous intersection code remains the same]
        if abs(np.cos(theta)) < 1e-10:
            x1 = x2 = x0
            y1 = np.sqrt(radius**2 - x0**2)
            y2 = -y1
            return [(x1, y1), (x2, y2)]
        
        m = np.tan(theta)
        c = y0 - m*x0
        
        a = 1 + m**2
        b = 2*m*c
        d = c**2 - radius**2
        
        discriminant = b**2 - 4*a*d
        if discriminant < 0:
            return None
        
        x1 = (-b + np.sqrt(discriminant))/(2*a)
        x2 = (-b - np.sqrt(discriminant))/(2*a)
        y1 = m*x1 + c
        y2 = m*x2 + c
        
        return [(x1, y1), (x2, y2)]

    @staticmethod
    def find_nearest_detector(x, y, detector_x, detector_y):
        distances = (detector_x - x)**2 + (detector_y - y)**2
        return np.argmin(distances)

    def angle_to_bin(self, theta):
        """Convert angle to bin index (0 to 179 degrees)"""
        # Normalize angle to [0, π)
        theta = theta % np.pi
        # Convert to bin index
        bin_idx = int((theta / np.pi) * self.n_angles)
        return min(bin_idx, self.n_angles - 1)  # Ensure we don't exceed bounds

    def simulate_batch(self, batch_params):
        """Simulate a batch of events"""
        n_events, seed = batch_params
        np.random.seed(seed)
        local_sinogram = np.zeros((self.n_detectors, self.n_angles))
        
        prob = self.phantom.flatten() / self.phantom.sum()
        
        for _ in range(n_events):
            # Generate random point
            idx = np.random.choice(len(prob), p=prob)
            y_idx, x_idx = np.unravel_index(idx, self.phantom.shape)
            
            x0 = (x_idx - self.image_size//2) * self.pixel_size
            y0 = (y_idx - self.image_size//2) * self.pixel_size
            
            # Generate random angle for first photon
            theta = np.random.uniform(0, 2*np.pi)
            
            intersections = self.line_circle_intersection(x0, y0, theta, self.detector_radius)
            if intersections is None:
                continue
                
            detector1 = self.find_nearest_detector(intersections[0][0], intersections[0][1], 
                                                 self.detector_x, self.detector_y)
            detector2 = self.find_nearest_detector(intersections[1][0], intersections[1][1], 
                                                 self.detector_x, self.detector_y)
            
            # Convert theta to angle bin (0-179 degrees)
            angle_bin = self.angle_to_bin(theta)
            
            # Record both detectors in sinogram
            local_sinogram[detector1, angle_bin] += 1
            local_sinogram[detector2, angle_bin] += 1
            
        return local_sinogram

    def simulate_events_parallel(self, n_events=200000, max_workers=None):
        start_time = time.time()
        
        if max_workers is None:
            max_workers = min(32, (n_events + 999) // 1000)
        
        batch_size = n_events // max_workers
        print(f"Running simulation with {max_workers} workers")
        print(f"Events per worker: {batch_size}")
        
        batches = [(batch_size, i) for i in range(max_workers)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(self.simulate_batch, batch): i 
                             for i, batch in enumerate(batches)}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    local_sinogram = future.result()
                    self.sinogram += local_sinogram
                    completed += 1
                    print(f"Completed batch {batch_idx+1}/{max_workers} ({completed/max_workers*100:.1f}%)")
                except Exception as e:
                    print(f"Batch {batch_idx} generated an exception: {e}")
        
        end_time = time.time()
        print(f"Total simulation time: {end_time - start_time:.2f} seconds")
    
    def plot_results(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot detector ring and phantom
        ax1.scatter(self.detector_x, self.detector_y, c='red', s=10, label='Detectors')
        im1 = ax1.imshow(self.phantom, extent=[-self.detector_radius, self.detector_radius, 
                                             -self.detector_radius, self.detector_radius])
        ax1.set_title('Phantom with Detector Ring')
        plt.colorbar(im1, ax=ax1)
        
        # Plot phantom alone
        im2 = ax2.imshow(self.phantom)
        ax2.set_title('Phantom (Activity Distribution)')
        plt.colorbar(im2, ax=ax2)
        
        # Plot sinogram with angle as x-axis and detector number as y-axis
        im3 = ax3.imshow(self.sinogram, aspect='auto', 
                        extent=[0, 180, 0, self.n_detectors],
                        origin='lower')
        ax3.set_title('Sinogram')
        ax3.set_xlabel('Angle (degrees)')
        ax3.set_ylabel('Detector Number')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.savefig('sinogram.png')
        plt.show()

if __name__ == '__main__':
    # Run simulation
    pet_sim = PETSimulation(detector_radius=100, n_detectors=180)
    pet_sim.simulate_events_parallel(n_events=2000000)
    pet_sim.plot_results()