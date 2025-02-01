# pet_simulator/simulator.py
import numpy as np
import multiprocessing as mp
from functools import partial
from .numba_utils import simulate_batch
from .utils import save_detector_lut

def process_batch(process_id: int, batch_size: int, shared_data: dict) -> np.ndarray:
    """
    Process a batch of events in a separate process.
    Use the process_id to seed the random number generator.
    """
    np.random.seed(process_id)
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

class PETSimulator:
    def __init__(self, geometry, image: np.ndarray, voxel_size: float):
        self.geometry = geometry
        self.image = image.astype(np.float64)
        self.voxel_size = voxel_size
        self._calculate_detector_positions()
        # Pre-calculate probability distribution for event generation
        total_activity = np.sum(self.image)
        assert total_activity > 0, "Total activity must be greater than 0."
        print('total_activity', total_activity)
        self.probabilities = self.image.ravel() / total_activity
        self.cumsum_prob = np.cumsum(self.probabilities)

    def _calculate_detector_positions(self):
        """Calculate detector positions based on geometry parameters."""
        angles = np.linspace(0, 2 * np.pi, self.geometry.crystals_per_ring, endpoint=False)
        self.detector_positions = np.zeros(
            (self.geometry.num_rings * self.geometry.crystals_per_ring, 3),
            dtype=np.float64
        )
        for ring in range(self.geometry.num_rings):
            z_pos = (ring - self.geometry.num_rings / 2) * self.geometry.crystal_axial_spacing
            start_idx = ring * self.geometry.crystals_per_ring
            end_idx = (ring + 1) * self.geometry.crystals_per_ring
            self.detector_positions[start_idx:end_idx, 0] = self.geometry.radius * np.cos(angles)
            self.detector_positions[start_idx:end_idx, 1] = self.geometry.radius * np.sin(angles)
            self.detector_positions[start_idx:end_idx, 2] = z_pos

    def simulate_events(self, num_events: int) -> np.ndarray:
        """Simulate events using multiprocessing."""
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
            process_batch_partial = partial(process_batch, batch_size=batch_size, shared_data=shared_data)
            results = pool.map(process_batch_partial, range(num_processes))
        total_events = np.vstack(results)
        return total_events

    def save_detector_positions(self, filename: str):
        """Save detector positions lookup table using the utils module."""
        save_detector_lut(filename, self.detector_positions)
