import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import concurrent.futures
import time
from scipy.stats import gaussian_kde
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

class PETSimulation:
    def __init__(self, detector_radius=100, n_detectors=180):
        self.detector_radius = detector_radius
        self.n_detectors = n_detectors
        
        # Create detector positions (equally spaced around the circle)
        self.detector_angles = np.linspace(0, 2*np.pi, n_detectors, endpoint=False)
        self.detector_x = detector_radius * np.cos(self.detector_angles)
        self.detector_y = detector_radius * np.sin(self.detector_angles)
        
        self.n_angles = 180
        self.sinogram = np.zeros((n_detectors, self.n_angles))
        
        self.image_size = 200
        self.pixel_size = 2 * detector_radius / self.image_size
        self.phantom = self.create_phantom()
        
        # Store events for animation
        self.events = []
        
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
        hot_spot1 = ((X - 20)**2 + (Y - 20)**2) <= 15**2
        phantom[hot_spot1] = 30
        
        hot_spot2 = ((X + 30)**2 + (Y + 30)**2) <= 20**2
        phantom[hot_spot2] = 20
        
        # Elliptical hot spot
        ellipse = ((X - 10)**2/400 + (Y + 20)**2/100) <= 1
        phantom[ellipse] = 40
        
        return phantom

    @staticmethod
    def line_circle_intersection(x0, y0, theta, radius):
        """Find intersection points of a line with the detector circle"""
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
        """Find the index of the nearest detector to point (x,y)"""
        distances = (detector_x - x)**2 + (detector_y - y)**2
        return np.argmin(distances)

    def angle_to_bin(self, theta):
        """Convert angle to bin index (0 to 179 degrees)"""
        theta = theta % np.pi
        bin_idx = int((theta / np.pi) * self.n_angles)
        return min(bin_idx, self.n_angles - 1)

    def generate_single_event(self):
        """Generate a single event for animation"""
        prob = self.phantom.flatten() / self.phantom.sum()
        
        # Generate random point
        idx = np.random.choice(len(prob), p=prob)
        y_idx, x_idx = np.unravel_index(idx, self.phantom.shape)
        
        x0 = (x_idx - self.image_size//2) * self.pixel_size
        y0 = (y_idx - self.image_size//2) * self.pixel_size
        
        # Generate random angle
        theta = np.random.uniform(0, 2*np.pi)
        
        intersections = self.line_circle_intersection(x0, y0, theta, self.detector_radius)
        if intersections is None:
            return None
            
        detector1 = self.find_nearest_detector(intersections[0][0], intersections[0][1], 
                                             self.detector_x, self.detector_y)
        detector2 = self.find_nearest_detector(intersections[1][0], intersections[1][1], 
                                             self.detector_x, self.detector_y)
        
        angle_bin = self.angle_to_bin(theta)
        
        return {
            'emission_point': (x0, y0),
            'intersections': intersections,
            'detectors': (detector1, detector2),
            'angle_bin': angle_bin
        }

    def create_animation(self, n_frames=500, interval=50):
        """Create an animation of the PET scanning process"""
        fig = plt.figure(figsize=(15, 5))
        
        # Create subplot layout
        gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])
        ax1 = fig.add_subplot(gs[0])  # Phantom with detector ring
        ax2 = fig.add_subplot(gs[1])  # Current event
        ax3 = fig.add_subplot(gs[2])  # Sinogram
        
        # Setup phantom and detector ring plot
        ax1.imshow(self.phantom, extent=[-self.detector_radius, self.detector_radius, 
                                    -self.detector_radius, self.detector_radius])
        ax1.scatter(self.detector_x, self.detector_y, c='red', s=10)
        ax1.set_title('Phantom')
        ax1.set_aspect('equal')
        
        # Setup event visualization
        detector_ring = Circle((0, 0), self.detector_radius, fill=False, color='black')
        ax2.add_patch(detector_ring)
        detector_points = ax2.scatter(self.detector_x, self.detector_y, c='red', s=10)
        ax2.set_xlim([-1.2*self.detector_radius, 1.2*self.detector_radius])
        ax2.set_ylim([-1.2*self.detector_radius, 1.2*self.detector_radius])
        ax2.set_title('Current Event')
        ax2.set_aspect('equal')
        
        # Setup sinogram
        sinogram_img = ax3.imshow(np.zeros((self.n_detectors, self.n_angles)), 
                                aspect='auto', extent=[0, 180, 0, self.n_detectors],
                                origin='lower')
        ax3.set_title('Sinogram')
        ax3.set_xlabel('Angle (degrees)')
        ax3.set_ylabel('Detector Number')
        
        # Animation storage
        self.animation_sinogram = np.zeros((self.n_detectors, self.n_angles))
        
        # Store emission points for density visualization
        emission_points_x = []
        emission_points_y = []
        
        # Create empty line objects for updating
        emission_point = ax2.scatter([], [], c='green', s=64)  # Current emission point
        emission_history = ax2.scatter([], [], c='green', s=20, alpha=0.2)  # Historical points
        active_detectors = ax2.scatter([], [], c='blue', s=100)
        photon_paths, = ax2.plot([], [], 'y-', alpha=0.5)
        
        def init():
            emission_point.set_offsets(np.empty((0, 2)))
            emission_history.set_offsets(np.empty((0, 2)))
            active_detectors.set_offsets(np.empty((0, 2)))
            photon_paths.set_data([], [])
            return [emission_point, emission_history, active_detectors, photon_paths, sinogram_img]

        def update(frame):
            # Generate new event
            event = self.generate_single_event()
            if event is None:
                return [emission_point, emission_history, active_detectors, photon_paths, sinogram_img]
            
            # Update emission point
            emission_x, emission_y = event['emission_point']
            emission_point.set_offsets([[emission_x, emission_y]])
            
            # Update emission history
            emission_points_x.append(emission_x)
            emission_points_y.append(emission_y)
            history_points = np.column_stack((emission_points_x, emission_points_y))
            emission_history.set_offsets(history_points)
            
            # Update color of historical points based on number of points
            n_points = len(emission_points_x)
            if n_points > 10:  # Only attempt density estimation with enough points
                try:
                    # Create a grid for density estimation
                    grid_size = 50
                    x_grid = np.linspace(min(emission_points_x), max(emission_points_x), grid_size)
                    y_grid = np.linspace(min(emission_points_y), max(emission_points_y), grid_size)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    
                    # Calculate point density using 2D histogram
                    H, _, _ = np.histogram2d(emission_points_x, emission_points_y, bins=(grid_size, grid_size),
                                        range=[[X.min(), X.max()], [Y.min(), Y.max()]])
                    
                    # Normalize density for coloring
                    H = H / H.max()
                    
                    # Interpolate density at point locations
                    from scipy.interpolate import interpn
                    points = (x_grid, y_grid)
                    density = interpn(points, H.T, history_points, method='linear', bounds_error=False, fill_value=0)
                    
                    # Update colors
                    emission_history.set_array(density)
                    emission_history.set_cmap('Greens')
                except Exception as e:
                    # Fallback to simple visualization if density estimation fails
                    emission_history.set_array(None)
                    emission_history.set_color('green')
                    emission_history.set_alpha(0.2)
            else:
                # Simple visualization for few points
                emission_history.set_array(None)
                emission_history.set_color('green')
                emission_history.set_alpha(0.2)
            
            # Update photon paths
            intersections = event['intersections']
            path_x = [intersections[0][0], emission_x, intersections[1][0]]
            path_y = [intersections[0][1], emission_y, intersections[1][1]]
            photon_paths.set_data(path_x, path_y)
            
            # Update active detectors
            detector1, detector2 = event['detectors']
            active_detector_positions = np.array([
                [self.detector_x[detector1], self.detector_y[detector1]],
                [self.detector_x[detector2], self.detector_y[detector2]]
            ])
            active_detectors.set_offsets(active_detector_positions)
            
            # Update sinogram
            angle_bin = event['angle_bin']
            self.animation_sinogram[detector1, angle_bin] += 1
            self.animation_sinogram[detector2, angle_bin] += 1
            sinogram_img.set_array(self.animation_sinogram)
            sinogram_img.set_clim(0, np.max(self.animation_sinogram))
            
            # Update frame counter
            ax2.set_title(f'Current Event (Frame {frame+1}/{n_frames})\nEmission Points: {n_points}')
            
            return [emission_point, emission_history, active_detectors, photon_paths, sinogram_img]

        # Create animation
        anim = animation.FuncAnimation(fig, update, init_func=init, frames=n_frames,
                                    interval=interval, blit=True)
        
        # Save animation with higher quality
        writer = animation.FFMpegWriter(fps=20, bitrate=3000)
        anim.save(f'pet_simulation{n_frames}.mp4', writer=writer)
        
        plt.close()


if __name__ == '__main__':
    # Create simulation and animation
    np.random.seed(42)  # For reproducibility
    pet_sim = PETSimulation(detector_radius=100, n_detectors=180)
    pet_sim.create_animation(n_frames=100, interval=50)