import numpy as np
import matplotlib.pyplot as plt

class PETSimulation:
    def __init__(self, detector_radius=100, n_detectors=360):
        self.detector_radius = detector_radius
        self.n_detectors = n_detectors
        
        # Create detector positions (equally spaced around the circle)
        self.detector_angles = np.linspace(0, 2*np.pi, n_detectors, endpoint=False)
        self.detector_x = detector_radius * np.cos(self.detector_angles)
        self.detector_y = detector_radius * np.sin(self.detector_angles)
        
        # Initialize sinogram
        self.sinogram = np.zeros((n_detectors//2, n_detectors))
        
        # Create density function (phantom)
        self.image_size = 200
        self.pixel_size = 2 * detector_radius / self.image_size
        self.phantom = self.create_phantom()
        
    def create_phantom(self):
        # Create a grid
        x = np.linspace(-self.detector_radius, self.detector_radius, self.image_size)
        y = np.linspace(-self.detector_radius, self.detector_radius, self.image_size)
        X, Y = np.meshgrid(x, y)
        
        phantom = np.zeros((self.image_size, self.image_size))
        
        # Create main circle
        main_circle = (X**2 + Y**2) <= (0.8 * self.detector_radius)**2
        phantom[main_circle] = 1
        
        # Add hot spots (more radioactive regions)
        # Circular hot spot 1
        hot_spot1 = ((X - 20)**2 + (Y - 20)**2) <= 15**2
        phantom[hot_spot1] = 3
        
        # Circular hot spot 2
        hot_spot2 = ((X + 30)**2 + (Y + 30)**2) <= 20**2
        phantom[hot_spot2] = 2
        
        # Elliptical hot spot
        ellipse = ((X - 10)**2/400 + (Y + 20)**2/100) <= 1
        phantom[ellipse] = 4
        
        return phantom
    
    def line_circle_intersection(self, x0, y0, theta):
        """
        Find intersection points of a line with the detector circle
        Line equation: y = mx + c where m = tan(theta)
        Circle equation: x² + y² = r²
        """
        # Handle vertical lines separately
        if abs(np.cos(theta)) < 1e-10:
            x1 = x2 = x0
            y1 = np.sqrt(self.detector_radius**2 - x0**2)
            y2 = -y1
            return [(x1, y1), (x2, y2)]
        
        m = np.tan(theta)
        c = y0 - m*x0
        
        # Quadratic equation coefficients: ax² + bx + d = 0
        a = 1 + m**2
        b = 2*m*c
        d = c**2 - self.detector_radius**2
        
        # Solve quadratic equation
        discriminant = b**2 - 4*a*d
        if discriminant < 0:
            return None
        
        x1 = (-b + np.sqrt(discriminant))/(2*a)
        x2 = (-b - np.sqrt(discriminant))/(2*a)
        y1 = m*x1 + c
        y2 = m*x2 + c
        
        return [(x1, y1), (x2, y2)]
    
    def find_nearest_detector(self, x, y):
        """Find the index of the nearest detector to point (x,y)"""
        distances = (self.detector_x - x)**2 + (self.detector_y - y)**2
        return np.argmin(distances)
    
    def simulate_events(self, n_events=20000):
        for _ in range(n_events):
            # Generate random point according to density function
            prob = self.phantom.flatten() / self.phantom.sum()
            idx = np.random.choice(len(prob), p=prob)
            y_idx, x_idx = np.unravel_index(idx, self.phantom.shape)
            
            # Convert index to coordinates
            x0 = (x_idx - self.image_size//2) * self.pixel_size
            y0 = (y_idx - self.image_size//2) * self.pixel_size
            
            # Generate random angle for first photon
            theta = np.random.uniform(0, 2*np.pi)
            
            # Find intersection points with detector ring
            intersections = self.line_circle_intersection(x0, y0, theta)
            if intersections is None:
                continue
                
            # Find nearest detectors for both intersection points
            detector1 = self.find_nearest_detector(*intersections[0])
            detector2 = self.find_nearest_detector(*intersections[1])
            
            # Record in sinogram
            sino_angle = min(detector1, detector2)
            sino_pos = max(detector1, detector2)
            self.sinogram[sino_angle % (self.n_detectors//2), sino_pos] += 1
    
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
        
        # Plot sinogram
        im3 = ax3.imshow(self.sinogram, aspect='auto')
        ax3.set_title('Sinogram')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.show()

# Run simulation
pet_sim = PETSimulation(detector_radius=100, n_detectors=180)
pet_sim.simulate_events(n_events=200000)
pet_sim.plot_results()