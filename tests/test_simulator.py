# tests/test_simulator.py
import unittest
import numpy as np
from pet_simulator.geometry import create_pet_geometry
from pet_simulator.simulator import PETSimulator

class TestPETSimulator(unittest.TestCase):
    def setUp(self):
        # Use a small test image for speed (e.g., 16x16x16) with a uniform density
        self.shape = (16, 16, 16)
        self.image = np.ones(self.shape, dtype=np.float64)
        self.voxel_size = 2.78

        # Use a test geometry (values similar to the main configuration)
        info = {
            'radius': np.float64(253.7067),
            'NrCrystalsPerRing': 288,
            'NrRings': 54,
            'crystalTransSpacing': np.float64(5.535),
            'crystalAxialSpacing': np.float64(6.6533),
            'crystalTransNr': 16,
            'crystalAxialNr': 9,
            'moduleAxialNr': 6,
            'moduleAxialSpacing': np.float64(89.82)
        }
        self.geometry = create_pet_geometry(info)
        self.simulator = PETSimulator(self.geometry, self.image, self.voxel_size)
        
        # Simulate a moderate number of events for testing
        self.num_events = int(1e6)
        self.events = self.simulator.simulate_events(self.num_events)
    
    def test_collinearity(self):
        # For each event, check that the vectors from the event to each detector are nearly opposite.
        # Tolerance: 1.5 degrees converted to radians.
        tol = np.deg2rad(2)
        for i in range(self.events.shape[0]):
            event_pos = self.events[i, 8:11]
            det1_pos = self.events[i, 2:5]
            det2_pos = self.events[i, 5:8]
            
            # Compute vectors from event to each detector
            v1 = det1_pos - event_pos
            v2 = det2_pos - event_pos
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue  # Skip degenerate cases
            
            u1 = v1 / norm1
            u2 = v2 / norm2
            
            # For collinearity the vectors should be opposite, so the dot product should be ~ -1.
            dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
            angle = np.arccos(dot)
            deviation = abs(np.pi - angle)
            
            self.assertTrue(deviation < tol,
                f"Event {i} not collinear: deviation = {deviation:.4f} rad exceeds tolerance {tol:.4f} rad")
    
    def test_distribution_alignment_histogram(self):
        """
        Test that the spatial distribution of simulated event positions aligns with
        the density distribution by comparing binned histograms using a Pearson correlation.
        This test uses a non-uniform (Gaussian) image so that the expected distribution
        has nonzero variance.
        """
        # Create a non-uniform Gaussian image.
        shape = self.shape
        # For the image, assume the first dimension is z, then y, then x.
        z = np.arange(shape[0]) - shape[0] / 2
        y = np.arange(shape[1]) - shape[1] / 2
        x = np.arange(shape[2]) - shape[2] / 2
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        sigma = 3.0
        gaussian = np.exp(-((X**2 + Y**2 + Z**2) / (2 * sigma**2)))
        nonuniform_image = gaussian.astype(np.float64)

        # Create a new simulator with the non-uniform image.
        simulator = PETSimulator(self.geometry, nonuniform_image, self.voxel_size)
        events = simulator.simulate_events(self.num_events)

        # Bin the simulated event positions back into a histogram on the voxel grid.
        hist = np.zeros(nonuniform_image.shape, dtype=np.int64)
        for event in events:
            # Convert physical coordinates back to voxel indices.
            # event position = (voxel_index - (shape/2)) * voxel_size.
            x_idx = int(np.floor(event[8] / self.voxel_size + nonuniform_image.shape[2] / 2))
            y_idx = int(np.floor(event[9] / self.voxel_size + nonuniform_image.shape[1] / 2))
            z_idx = int(np.floor(event[10] / self.voxel_size + nonuniform_image.shape[0] / 2))
            if (0 <= x_idx < nonuniform_image.shape[2] and
                0 <= y_idx < nonuniform_image.shape[1] and
                0 <= z_idx < nonuniform_image.shape[0]):
                hist[z_idx, y_idx, x_idx] += 1

        # Normalize both the histogram and the expected image to form probability distributions.
        hist_prob = hist / np.sum(hist)
        expected_prob = nonuniform_image / np.sum(nonuniform_image)

        # Flatten the distributions and compute the Pearson correlation coefficient.
        hist_prob_flat = hist_prob.flatten()
        expected_prob_flat = expected_prob.flatten()
        corr_matrix = np.corrcoef(hist_prob_flat, expected_prob_flat)
        corr = corr_matrix[0, 1]
        self.assertTrue(corr > 0.99, f"Histogram correlation is low: {corr:.4f}")
        
    def test_event_fov(self):
        """
        Verify that every simulated event's z coordinate lies within the expected Field-of-View (FOV).
        The expected FOV in z is determined by:
            FOV_z = (num_rings * crystal_axial_spacing) / 2
        """
        expected_fov = (self.geometry.num_rings * self.geometry.crystal_axial_spacing) / 2.0
        # The event z positions are in column 10.
        event_z = self.events[:, 10]
        # Check that every event z coordinate (absolute value) is less than expected_fov.
        self.assertTrue(np.all(np.abs(event_z) < expected_fov),
                        f"Some event z positions exceed the expected FOV of ±{expected_fov:.2f}.")



if __name__ == '__main__':
    unittest.main()
