import numpy as np
from dataclasses import dataclass
import math
from numba import jit, njit, prange
from numba.typed import List
from typing import Tuple, Optional


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

@njit
def manual_unravel_index(idx: int, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Manual implementation of np.unravel_index for 3D arrays"""
    # For a 3D array with shape (d1, d2, d3)
    d1, d2, d3 = shape
    
    # Calculate indices
    z = idx // (d2 * d3)  # First dimension
    remainder = idx % (d2 * d3)
    y = remainder // d3   # Second dimension
    x = remainder % d3    # Third dimension
    
    return z, y, x

@njit
def find_detector_intersection(pos: np.ndarray, direction: np.ndarray, 
                             detector_positions: np.ndarray, radius: float,
                             num_rings: float, crystal_axial_spacing: float) -> Optional[int]:
    """Numba optimized detector intersection calculation"""
    # Calculate intersection with detector cylinder
    a = direction[0]**2 + direction[1]**2
    b = 2 * (pos[0]*direction[0] + pos[1]*direction[1])
    c = pos[0]**2 + pos[1]**2 - radius**2
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return -1  # Changed from None to -1 for Numba compatibility
        
    # Calculate intersection point
    t = (-b + np.sqrt(discriminant)) / (2*a)
    intersection = pos + t * direction
    
    # Find nearest detector using manual distance calculation
    min_dist = np.inf
    nearest_detector = -1
    for i in range(len(detector_positions)):
        dist = (detector_positions[i,0] - intersection[0])**2 + \
               (detector_positions[i,1] - intersection[1])**2 + \
               (detector_positions[i,2] - intersection[2])**2
        if dist < min_dist:
            min_dist = dist
            nearest_detector = i
    
    # Check if intersection is within axial FOV
    if abs(intersection[2]) > (num_rings * crystal_axial_spacing / 2):
        return -1  # Changed from None to -1 for Numba compatibility
        
    return nearest_detector

@njit
def simulate_single_batch(start_idx: int, end_idx: int, image: np.ndarray, 
                         shape: Tuple[int, int, int], voxel_size: float,
                         detector_positions: np.ndarray, radius: float,
                         num_rings: int, crystal_axial_spacing: float,
                         cumsum_prob: np.ndarray) -> np.ndarray:
    """Simulate a batch of events"""
    # Pre-allocate arrays for this batch
    batch_size = end_idx - start_idx
    events = np.zeros((batch_size, 2), dtype=np.int64)
    valid_count = 0
    
    for _ in range(batch_size):
        # Generate random annihilation point
        rand_val = np.random.random()
        idx = np.searchsorted(cumsum_prob, rand_val)
        z, y, x = manual_unravel_index(idx, shape)
        
        # Convert to physical coordinates
        x_pos = (x - shape[2]/2) * voxel_size
        y_pos = (y - shape[1]/2) * voxel_size
        z_pos = (z - shape[0]/2) * voxel_size
        
        # Create position array
        pos = np.array([x_pos, y_pos, z_pos], dtype=np.float64)
        
        # Generate random direction
        phi = np.random.uniform(0, 2*np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        # Direction vectors
        dir1 = np.array([sin_theta*np.cos(phi), 
                        sin_theta*np.sin(phi), 
                        cos_theta], dtype=np.float64)
        dir2 = -dir1
        
        # Find intersections
        det1 = find_detector_intersection(pos, dir1, detector_positions, radius, 
                                        num_rings, crystal_axial_spacing)
        if det1 >= 0:
            det2 = find_detector_intersection(pos, dir2, detector_positions, radius, 
                                            num_rings, crystal_axial_spacing)
            if det2 >= 0:
                events[valid_count, 0] = det1
                events[valid_count, 1] = det2
                valid_count += 1
    
    return events[:valid_count]

@njit(parallel=True)
def simulate_events_batch(num_events: int, image: np.ndarray, voxel_size: float,
                         detector_positions: np.ndarray, radius: float,
                         num_rings: int, crystal_axial_spacing: float) -> np.ndarray:
    """Numba optimized parallel batch event simulation"""
    # Calculate probabilities once
    total_activity = np.sum(image)
    probabilities = image.ravel() / total_activity
    cumsum_prob = np.cumsum(probabilities)
    
    # Store shape for unraveling
    shape = (image.shape[0], image.shape[1], image.shape[2])
    
    # Determine number of parallel batches
    num_threads = min(num_events, 32)  # Limit max number of threads
    batch_size = num_events // num_threads
    remainders = num_events % num_threads
    
    # Pre-allocate list to store results from each thread
    all_events = List()
    for i in range(num_threads):
        # Calculate batch size for this thread
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        if i == num_threads - 1:
            end_idx += remainders
            
        # Simulate batch and store results
        batch_events = simulate_single_batch(
            start_idx, end_idx, image, shape, voxel_size,
            detector_positions, radius, num_rings,
            crystal_axial_spacing, cumsum_prob
        )
        all_events.append(batch_events)
    
    # Count total valid events
    total_valid = 0
    for batch in all_events:
        total_valid += len(batch)
    
    # Combine all valid events
    combined_events = np.zeros((total_valid, 2), dtype=np.int64)
    current_idx = 0
    for batch in all_events:
        batch_size = len(batch)
        combined_events[current_idx:current_idx + batch_size] = batch
        current_idx += batch_size
    
    return combined_events

# Rest of the code remains the same
class PETSimulator:
    def __init__(self, geometry: PETGeometry, image: np.ndarray, voxel_size: float):
        self.geometry = geometry
        self.image = image.astype(np.float64)  # Ensure float64 type
        self.voxel_size = voxel_size
        self._calculate_detector_positions()
    
    def _calculate_detector_positions(self):
        """Calculate detector positions"""
        angles = np.linspace(0, 2*np.pi, self.geometry.crystals_per_ring, endpoint=False)
        self.detector_positions = np.zeros((self.geometry.num_rings * 
                                          self.geometry.crystals_per_ring, 3), 
                                         dtype=np.float64)  # Ensure float64 type
        
        for ring in range(self.geometry.num_rings):
            z_pos = (ring - self.geometry.num_rings/2) * self.geometry.crystal_axial_spacing
            start_idx = ring * self.geometry.crystals_per_ring
            end_idx = (ring + 1) * self.geometry.crystals_per_ring
            
            self.detector_positions[start_idx:end_idx, 0] = self.geometry.radius * np.cos(angles)
            self.detector_positions[start_idx:end_idx, 1] = self.geometry.radius * np.sin(angles)
            self.detector_positions[start_idx:end_idx, 2] = z_pos
    
    def simulate_events(self, num_events: int) -> np.ndarray:
        """Main simulation interface using optimized batch processing"""
        return simulate_events_batch(
            num_events,
            self.image,
            self.voxel_size,
            self.detector_positions,
            self.geometry.radius,
            self.geometry.num_rings,
            self.geometry.crystal_axial_spacing
        )

def main():
    import time
    
    # Load 3D image
    image = np.load('3d_image_2.npy')
    
    # Create geometry from info
    geometry = create_pet_geometry(info)
    
    # Create simulator
    simulator = PETSimulator(geometry, image, voxel_size=2.78)
    
    # Compile functions by running a small batch first
    print("Compiling functions...")
    _ = simulator.simulate_events(100)
    
    # Simulate actual events with timing
    print("Starting main simulation...")
    start_time = time.time()
    events = simulator.simulate_events(num_events=2000000)
    end_time = time.time()
    
    # Save events
    np.savetxt("listmode_data_claude_numba.txt", events, fmt="%d")
    
    print(f"Generated {len(events)} valid events")
    print(f"Simulation time: {end_time - start_time:.2f} seconds")
    print("Events shape:", events.shape)
    print("Sample events:\n", events[:5])

if __name__ == "__main__":
    main()