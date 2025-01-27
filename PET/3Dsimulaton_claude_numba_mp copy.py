import numpy as np
from dataclasses import dataclass
import math
from numba import njit
from typing import Tuple, Optional
import multiprocessing as mp
from functools import partial


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
    
    
def save_detector_lut(filename: str, detector_positions: np.ndarray):
    """
    Save detector lookup table (LUT)
    Args:
        filename: output file name
        detector_positions: array with detector positions
    """
    # Create detector IDs array
    detector_ids = np.arange(len(detector_positions))
    
    # Combine detector IDs with their positions
    lut_data = np.column_stack((detector_ids, detector_positions))
    
    # Save with header and format
    header = "detector_id x y z"
    fmt = ['%d'] + ['%.6f']*3
    np.savetxt(filename, lut_data, fmt=fmt, header=header)

@njit
def manual_unravel_index(idx: int, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Manual implementation of np.unravel_index for 3D arrays"""
    d1, d2, d3 = shape
    z = idx // (d2 * d3)
    remainder = idx % (d2 * d3)
    y = remainder // d3
    x = remainder % d3
    return z, y, x


@njit
def find_detector_intersection(pos: np.ndarray, direction: np.ndarray,
                               detector_positions: np.ndarray, radius: float,
                               num_rings: float, crystal_axial_spacing: float) -> int:
    """Numba optimized detector intersection calculation"""
    a = direction[0]**2 + direction[1]**2
    b = 2 * (pos[0]*direction[0] + pos[1]*direction[1])
    c = pos[0]**2 + pos[1]**2 - radius**2

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return -1

    t = (-b + np.sqrt(discriminant)) / (2*a)
    intersection = pos + t * direction

    min_dist = np.inf
    nearest_detector = -1
    
    if abs(intersection[2]) > (num_rings * crystal_axial_spacing / 2):
        return -1
    
    for i in range(len(detector_positions)):
        dist = (detector_positions[i, 0] - intersection[0])**2 + \
               (detector_positions[i, 1] - intersection[1])**2 + \
               (detector_positions[i, 2] - intersection[2])**2
        if dist < min_dist:
            min_dist = dist
            nearest_detector = i

    return nearest_detector


@njit
def simulate_batch(batch_size: int, image: np.ndarray, shape: Tuple[int, int, int],
                   voxel_size: float, detector_positions: np.ndarray, radius: float,
                   num_rings: int, crystal_axial_spacing: float,
                   cumsum_prob: np.ndarray) -> np.ndarray:
    """
    Simulate a batch of events
    Returns: array with shape (N, 11) containing:
        - detector1_id, detector2_id (columns 0-1)
        - detector1_x, detector1_y, detector1_z (columns 2-4)
        - detector2_x, detector2_y, detector2_z (columns 5-7)
        - event_x, event_y, event_z (columns 8-10)
    """
    # Pre-allocate with more columns for additional information
    events = np.zeros((batch_size, 11), dtype=np.float64)
    valid_count = 0

    for _ in range(batch_size):
        # Generate random annihilation point
        rand_val = np.random.random()
        idx = np.searchsorted(cumsum_prob, rand_val)
        z, y, x = manual_unravel_index(idx, shape)

        # Convert to physical coordinates for event position
        x_pos = (x - shape[2]/2) * voxel_size
        y_pos = (y - shape[1]/2) * voxel_size
        z_pos = (z - shape[0]/2) * voxel_size
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
                # Store detector IDs
                events[valid_count, 0] = det1
                events[valid_count, 1] = det2
                
                # Store first detector position
                events[valid_count, 2:5] = detector_positions[det1]
                
                # Store second detector position
                events[valid_count, 5:8] = detector_positions[det2]
                
                # Store event position
                events[valid_count, 8:11] = pos
                
                valid_count += 1

    return events[:valid_count]


def save_events(filename: str, events: np.ndarray, save_full_data: bool = False):
    """
    Save events with option for full or minimal data
    Args:
        filename: output file name
        events: array with event data
        save_full_data: if True, save all data including positions
    """
    if save_full_data:
        # Save full data: detector IDs, positions and event position
        header = ("det1_id det2_id "
                 "det1_x det1_y det1_z "
                 "det2_x det2_y det2_z "
                 "event_x event_y event_z")
        fmt = ['%d', '%d'] + ['%.6f']*9
        np.savetxt(filename, events, fmt=fmt, header=header)
    else:
        # Save only detector IDs
        header = "det1_id det2_id"
        detector_ids = events[:, :2]  # Take only first two columns
        np.savetxt(filename, detector_ids, fmt='%d', header=header)



def process_batch(process_id: int, batch_size: int, shared_data: dict) -> np.ndarray:
    """Process a batch of events in a separate process"""
    np.random.seed(
        process_id)  # Ensure different random numbers for each process
    return simulate_batch(
        batch_size,
        shared_data['image'],
        shared_data['shape'],
        shared_data['voxel_size'],
        shared_data['detector_positions'],
        shared_data['radius'],
        shared_data['num_rings'],
        shared_data['crystal_axial_spacing'],
        shared_data['cumsum_prob']
    )


# Rest of the code remains the same
class PETSimulator:
    def __init__(self, geometry: PETGeometry, image: np.ndarray, voxel_size: float):
        self.geometry = geometry
        self.image = image.astype(np.float64)
        self.voxel_size = voxel_size
        self._calculate_detector_positions()

        # Pre-calculate probability distribution
        total_activity = np.sum(self.image)
        self.probabilities = self.image.ravel() / total_activity
        self.cumsum_prob = np.cumsum(self.probabilities)

    def _calculate_detector_positions(self):
        """Calculate detector positions"""
        angles = np.linspace(0, 2*np.pi, self.geometry.crystals_per_ring, endpoint=False)
        self.detector_positions = np.zeros((self.geometry.num_rings * 
                                          self.geometry.crystals_per_ring, 3),
                                         dtype=np.float64)
        
        for ring in range(self.geometry.num_rings):
            z_pos = (ring - self.geometry.num_rings/2) * self.geometry.crystal_axial_spacing
            start_idx = ring * self.geometry.crystals_per_ring
            end_idx = (ring + 1) * self.geometry.crystals_per_ring
            
            self.detector_positions[start_idx:end_idx, 0] = self.geometry.radius * np.cos(angles)
            self.detector_positions[start_idx:end_idx, 1] = self.geometry.radius * np.sin(angles)
            self.detector_positions[start_idx:end_idx, 2] = z_pos

    def simulate_events(self, num_events: int) -> np.ndarray:
        """Simulate events using multiprocessing"""
        shared_data = {
            'image': self.image,
            'shape': self.image.shape,
            'voxel_size': self.voxel_size,
            'detector_positions': self.detector_positions,
            'radius': self.geometry.radius,
            'num_rings': self.geometry.num_rings,
            'crystal_axial_spacing': self.geometry.crystal_axial_spacing,
            'cumsum_prob': self.cumsum_prob
        }

        num_processes = mp.cpu_count()
        batch_size = num_events // num_processes

        with mp.Pool(num_processes) as pool:
            process_batch_partial = partial(process_batch,
                                         batch_size=batch_size,
                                         shared_data=shared_data)
            results = pool.map(process_batch_partial, range(num_processes))

        total_events = np.vstack(results)
        return total_events
    
    def save_detector_positions(self, filename: str):
        """Save detector positions lookup table"""
        save_detector_lut(filename, self.detector_positions)

def main():
    import time
    
    # Load 3D image
    image = np.load('3d_image_2.npy')
    
    # Create geometry from info
    geometry = create_pet_geometry(info)
    
    # Create simulator
    simulator = PETSimulator(geometry, image, voxel_size=2.78)
    
    # Save detector LUT first
    simulator.save_detector_positions("detector_lut.txt")
    
    # Simulate events with timing
    print("Starting simulation...")
    start_time = time.time()
    events = simulator.simulate_events(num_events=int(1e8))
    end_time = time.time()
    
    # Save events in both formats
    save_events("listmode_data_minimal.txt", events, save_full_data=False)
    save_events("listmode_data_full.txt", events, save_full_data=True)
    
    print(f"Generated {len(events)} valid events")
    print(f"Simulation time: {end_time - start_time:.2f} seconds")
    print("Events shape:", events.shape)
    print("\nSample event (with positions):")
    print("Detector 1 ID:", events[0, 0])
    print("Detector 2 ID:", events[0, 1])
    print("Detector 1 position (x,y,z):", events[0, 2:5])
    print("Detector 2 position (x,y,z):", events[0, 5:8])
    print("Event position (x,y,z):", events[0, 8:11])

if __name__ == "__main__":
    main()
