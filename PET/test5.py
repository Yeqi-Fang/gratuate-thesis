import torch
import numpy as np
import matplotlib.pyplot as plt

class PETSimulation:
    def __init__(self, detector_radius=100, n_detectors=360):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.detector_radius = detector_radius
        self.n_detectors = n_detectors
        
        self.detector_angles = torch.linspace(0, 2*torch.pi, n_detectors, device=self.device)
        self.detector_x = detector_radius * torch.cos(self.detector_angles)
        self.detector_y = detector_radius * torch.sin(self.detector_angles)
        
        self.n_angles = 180
        self.sinogram = torch.zeros((self.n_angles, n_detectors), device=self.device)
        
        self.image_size = 200
        self.pixel_size = 2 * detector_radius / self.image_size
        self.phantom = self.create_phantom()
        
    def create_phantom(self):
        x = torch.linspace(-self.detector_radius, self.detector_radius, self.image_size, device=self.device)
        y = torch.linspace(-self.detector_radius, self.detector_radius, self.image_size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        phantom = torch.zeros((self.image_size, self.image_size), device=self.device)
        
        main_circle = (X**2 + Y**2) <= (0.8 * self.detector_radius)**2
        phantom[main_circle] = 1.0
        
        hot_spot1 = ((X - 20)**2 + (Y - 20)**2) <= 15**2
        phantom[hot_spot1] = 3.0
        
        hot_spot2 = ((X + 30)**2 + (Y + 30)**2) <= 20**2
        phantom[hot_spot2] = 2.0
        
        ellipse = ((X - 10)**2/400 + (Y + 20)**2/100) <= 1
        phantom[ellipse] = 4.0
        
        return phantom
    
    def line_circle_intersection(self, x0, y0, theta):
        """
        Find intersection points of a line with the detector circle using vectorized operations
        Returns: tensor of shape (2, 2) containing the two intersection points
        """
        # Handle nearly vertical lines
        if abs(torch.cos(theta)) < 1e-10:
            x1 = x2 = x0
            y1 = torch.sqrt(self.detector_radius**2 - x0**2)
            y2 = -y1
            return torch.stack([
                torch.tensor([x1, y1], device=self.device),
                torch.tensor([x2, y2], device=self.device)
            ])
        
        m = torch.tan(theta)
        c = y0 - m*x0
        
        a = 1 + m**2
        b = 2*m*c
        d = c**2 - self.detector_radius**2
        
        discriminant = b**2 - 4*a*d
        if discriminant < 0:
            return None
        
        x1 = (-b + torch.sqrt(discriminant))/(2*a)
        x2 = (-b - torch.sqrt(discriminant))/(2*a)
        y1 = m*x1 + c
        y2 = m*x2 + c
        
        return torch.stack([
            torch.tensor([x1, y1], device=self.device),
            torch.tensor([x2, y2], device=self.device)
        ])
    
    def find_nearest_detector(self, points):
        """
        Find nearest detector for each intersection point
        points: tensor of shape (2, 2) containing (x, y) coordinates
        """
        distances = (self.detector_x.unsqueeze(0) - points[:, 0].unsqueeze(1))**2 + \
                   (self.detector_y.unsqueeze(0) - points[:, 1].unsqueeze(1))**2
        return torch.argmin(distances, dim=1)
    
    def angle_to_bin(self, theta):
        theta = theta % torch.pi
        bin_idx = (theta / torch.pi * self.n_angles).long()
        return bin_idx
    
    @torch.no_grad()
    def simulate_events(self, n_events=20000):
        print("Simulating events...")
        prob = self.phantom.flatten() / self.phantom.sum()
        
        batch_size = 1000
        for batch_start in range(0, n_events, batch_size):
            current_batch_size = min(batch_size, n_events - batch_start)
            
            # Generate random points
            indices = torch.multinomial(prob, current_batch_size, replacement=True)
            y_idx = torch.div(indices, self.image_size, rounding_mode='floor')
            x_idx = indices % self.image_size
            
            # Convert indices to coordinates
            x0 = (x_idx.float() - self.image_size/2) * self.pixel_size
            y0 = (y_idx.float() - self.image_size/2) * self.pixel_size
            
            # Generate random angles
            thetas = torch.rand(current_batch_size, device=self.device) * 2 * torch.pi
            
            for i in range(current_batch_size):
                intersections = self.line_circle_intersection(x0[i], y0[i], thetas[i])
                if intersections is None:
                    continue
                
                detector_indices = self.find_nearest_detector(intersections)
                angle_bin = self.angle_to_bin(thetas[i])
                
                self.sinogram[angle_bin, detector_indices[0]] += 1
                self.sinogram[angle_bin, detector_indices[1]] += 1
            
            if batch_start % 5000 == 0:
                print(f"Processed {batch_start + current_batch_size}/{n_events} events")
    
    def plot_results(self):
        phantom_cpu = self.phantom.cpu().numpy()
        sinogram_cpu = self.sinogram.cpu().numpy()
        detector_x_cpu = self.detector_x.cpu().numpy()
        detector_y_cpu = self.detector_y.cpu().numpy()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot detector ring and phantom
        ax1.scatter(detector_x_cpu, detector_y_cpu, c='red', s=10, label='Detectors')
        im1 = ax1.imshow(phantom_cpu, extent=[-self.detector_radius, self.detector_radius, 
                                            -self.detector_radius, self.detector_radius])
        ax1.set_title('Phantom with Detector Ring')
        plt.colorbar(im1, ax=ax1)
        
        # Plot phantom alone
        im2 = ax2.imshow(phantom_cpu)
        ax2.set_title('Phantom (Activity Distribution)')
        plt.colorbar(im2, ax=ax2)
        
        # Plot sinogram
        im3 = ax3.imshow(sinogram_cpu, aspect='auto', 
                        extent=[0, self.n_detectors, 0, 180],
                        origin='lower')
        ax3.set_title('Sinogram')
        ax3.set_xlabel('Detector Number')
        ax3.set_ylabel('Angle (degrees)')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.show()

# Run simulation
pet_sim = PETSimulation(detector_radius=100, n_detectors=360)
pet_sim.simulate_events(n_events=200000)
pet_sim.plot_results()