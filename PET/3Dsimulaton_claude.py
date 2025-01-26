import numpy as np
from dataclasses import dataclass
import math

info = {'min_rsector_difference': np.float64(0.0),
        'crystal_length': np.float64(0.0),
        'radius': np.float64(253.7067),
        'crystalTransNr': 16,
        'crystalTransSpacing': np.float64(5.535),
        'crystalAxialNr': 9,
        'crystalAxialSpacing': np.float64(6.6533),
        'submoduleAxialNr': 1,
        'submoduleAxialSpacing': np.float64(0.0),
        'submoduleTransNr': 1,
        'submoduleTransSpacing': np.float64(0.0),
        'moduleTransNr': 1,
        'moduleTransSpacing': np.float64(0.0),
        'moduleAxialNr': 6,
        'moduleAxialSpacing': np.float64(89.82),
        'rsectorTransNr': 18,
        'rsectorAxialNr': 1,
        'TOF': 1,
        'num_tof_bins': np.float64(29.0),
        'tof_range': np.float64(735.7705),
        'tof_fwhm': np.float64(57.71),
        'NrCrystalsPerRing': 288,
        'NrRings': 54,
        'firstCrystalAxis': 0}


@dataclass
class PETGeometry:
    """PET scanner geometry parameters"""
    radius: float
    crystals_per_ring: int
    num_rings: int
    crystal_trans_spacing: float
    crystal_axial_spacing: float
    crystal_trans_nr: int
    crystal_axial_nr: int
    module_axial_nr: int
    module_axial_spacing: float

class PETSimulator:
    def __init__(self, geometry: PETGeometry, image: np.ndarray, voxel_size: float):
        """
        Initialize PET simulator
        
        Args:
            geometry: PET scanner geometry parameters
            image: 3D activity distribution (128x128x128)
            voxel_size: Size of each voxel in mm
        """
        self.geometry = geometry
        self.image = image
        self.voxel_size = voxel_size
        
        # Calculate detector positions
        self._calculate_detector_positions()
        
    def _calculate_detector_positions(self):
        """Calculate the positions of all detectors in 3D space"""
        # Calculate angular positions around the ring
        angles = np.linspace(0, 2*np.pi, self.geometry.crystals_per_ring, endpoint=False)
        
        # Initialize arrays for detector positions
        self.detector_positions = np.zeros((self.geometry.num_rings * 
                                          self.geometry.crystals_per_ring, 3))
        
        # Calculate positions for each ring
        for ring in range(self.geometry.num_rings):
            z_pos = (ring - self.geometry.num_rings/2) * self.geometry.crystal_axial_spacing
            start_idx = ring * self.geometry.crystals_per_ring
            end_idx = (ring + 1) * self.geometry.crystals_per_ring
            
            # Set x, y positions for this ring
            self.detector_positions[start_idx:end_idx, 0] = self.geometry.radius * np.cos(angles)
            self.detector_positions[start_idx:end_idx, 1] = self.geometry.radius * np.sin(angles)
            self.detector_positions[start_idx:end_idx, 2] = z_pos
            
    def simulate_events(self, num_events: int) -> np.ndarray:
        """
        Simulate PET events
        
        Args:
            num_events: Number of events to simulate
            
        Returns:
            Array of shape (N, 2) containing detector pairs for valid events
        """
        events = []
        
        # Calculate total activity and probabilities
        total_activity = np.sum(self.image)
        probabilities = self.image.flatten() / total_activity
        
        for _ in range(num_events):
            # Generate random annihilation point based on activity distribution
            idx = np.random.choice(len(probabilities), p=probabilities)
            z, y, x = np.unravel_index(idx, self.image.shape)
            
            # Convert to physical coordinates (mm)
            x_pos = (x - self.image.shape[2]/2) * self.voxel_size
            y_pos = (y - self.image.shape[1]/2) * self.voxel_size
            z_pos = (z - self.image.shape[0]/2) * self.voxel_size
            
            # Generate random direction for photon pair
            phi = np.random.uniform(0, 2*np.pi)
            cos_theta = np.random.uniform(-1, 1)
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            # Direction vectors for the two photons
            dir1 = np.array([sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta])
            dir2 = -dir1
            
            # Find intersections with detector cylinder
            det1 = self._find_detector_intersection(np.array([x_pos, y_pos, z_pos]), dir1)
            det2 = self._find_detector_intersection(np.array([x_pos, y_pos, z_pos]), dir2)
            
            if det1 is not None and det2 is not None:
                events.append([det1, det2])
                
        return np.array(events)
    
    def _find_detector_intersection(self, pos: np.ndarray, direction: np.ndarray) -> int:
        """Find the detector ID that intersects with the given line of response"""
        # Calculate intersection with detector cylinder
        a = direction[0]**2 + direction[1]**2
        b = 2 * (pos[0]*direction[0] + pos[1]*direction[1])
        c = pos[0]**2 + pos[1]**2 - self.geometry.radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
            
        # Calculate intersection point
        t = (-b + np.sqrt(discriminant)) / (2*a)
        intersection = pos + t * direction
        
        # Find nearest detector
        distances = np.sum((self.detector_positions - intersection)**2, axis=1)
        nearest_detector = np.argmin(distances)
        
        # Check if intersection is within axial FOV
        if abs(intersection[2]) > (self.geometry.num_rings * 
                                 self.geometry.crystal_axial_spacing / 2):
            return None
            
        return nearest_detector

def create_pet_geometry(info: dict) -> PETGeometry:
    """Create PETGeometry from info dictionary"""
    return PETGeometry(
        radius=float(info['radius']),
        crystals_per_ring=int(info['NrCrystalsPerRing']),
        num_rings=int(info['NrRings']),
        crystal_trans_spacing=float(info['crystalTransSpacing']),
        crystal_axial_spacing=float(info['crystalAxialSpacing']),
        crystal_trans_nr=int(info['crystalTransNr']),
        crystal_axial_nr=int(info['crystalAxialNr']),
        module_axial_nr=int(info['moduleAxialNr']),
        module_axial_spacing=float(info['moduleAxialSpacing'])
    )

def main():
    # Load 3D image
    image = np.load('3d_image_2.npy')
    
    # Create geometry from info
    geometry = create_pet_geometry(info)
    
    # Create simulator
    simulator = PETSimulator(geometry, image, voxel_size=2.78)
    
    # Simulate events
    events = simulator.simulate_events(num_events=2000)
    
    # Save events
    # np.save('pet_events.npy', events)
    np.savetxt("listmode_data_claude.txt", events, fmt="%d")
    
    print(f"Generated {len(events)} valid events")
    print("Events shape:", events.shape)
    print("Sample events:\n", events[:5])

if __name__ == "__main__":
    main()