# pet_simulator/numba_utils.py
import numpy as np
from numba import njit
from typing import Tuple

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
        dist = ((detector_positions[i, 0] - intersection[0])**2 +
                (detector_positions[i, 1] - intersection[1])**2 +
                (detector_positions[i, 2] - intersection[2])**2)
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
    Simulate a batch of events.
    Returns an array with shape (N, 11) containing:
      - detector1_id, detector2_id (columns 0-1)
      - detector1_x, detector1_y, detector1_z (columns 2-4)
      - detector2_x, detector2_y, detector2_z (columns 5-7)
      - event_x, event_y, event_z (columns 8-10)
    """
    events = np.zeros((batch_size, 11), dtype=np.float64)
    valid_count = 0

    for _ in range(batch_size):
        # Choose a random annihilation point based on the cumulative probability
        rand_val = np.random.random()
        idx = np.searchsorted(cumsum_prob, rand_val)
        z, y, x = manual_unravel_index(idx, shape)

        # Convert voxel indices to physical coordinates
        x_pos = (x - shape[2] / 2) * voxel_size
        y_pos = (y - shape[1] / 2) * voxel_size
        z_pos = (z - shape[0] / 2) * voxel_size
        pos = np.array([x_pos, y_pos, z_pos], dtype=np.float64)

        # Generate random direction
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        dir1 = np.array([sin_theta * np.cos(phi),
                         sin_theta * np.sin(phi),
                         cos_theta], dtype=np.float64)
        dir2 = -dir1

        # Find intersections with the detector ring
        det1 = find_detector_intersection(pos, dir1, detector_positions, radius,
                                          num_rings, crystal_axial_spacing)
        if det1 >= 0:
            det2 = find_detector_intersection(pos, dir2, detector_positions, radius,
                                              num_rings, crystal_axial_spacing)
            if det2 >= 0:
                events[valid_count, 0] = det1
                events[valid_count, 1] = det2
                events[valid_count, 2:5] = detector_positions[det1]
                events[valid_count, 5:8] = detector_positions[det2]
                events[valid_count, 8:11] = pos
                valid_count += 1

    return events[:valid_count]
