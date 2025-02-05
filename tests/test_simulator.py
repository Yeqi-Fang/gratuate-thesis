# tests/test_simulator.py
import unittest
import numpy as np
from pet_simulator.geometry import create_pet_geometry
from pet_simulator.simulator import PETSimulator

class TestPETSimulator(unittest.TestCase):
    def setUp(self):
        # Use a small test image for speed (e.g., 16x16x16) with a uniform density
        self.shape = (128, 128, 128)
        # self.image = np.random.rand(*self.shape).astype(np.float32)
        self.image = np.ones(self.shape, dtype=np.float32)
        # self.image = np.ones(self.shape, dtype=np.float32)
        self.voxel_size = 2.78

        # Use a test geometry (values similar to the main configuration)
        info = {
            'radius': np.float32(290.56),
            'NrCrystalsPerRing': 544,
            'NrRings': 68,
            'crystalTransSpacing': np.float32(4.03125),
            'crystalAxialSpacing': np.float32(5.31556),
            'crystalTransNr': 16,
            'crystalAxialNr': 9,
            'moduleAxialNr': 6,
            'moduleAxialSpacing': np.float32(89.82)
        }
        self.geometry = create_pet_geometry(info)
        self.simulator = PETSimulator(self.geometry, self.image, self.voxel_size)
        
        # Simulate a moderate number of events for testing
        self.num_events = int(1e6)
        self.events = self.simulator.simulate_events(self.num_events)
    
    def test_collinearity(self):
        # For each event, check that the vectors from the event to each detector are nearly opposite.
        # Tolerance: 1.5 degrees converted to radians.
        tol = np.deg2rad(10)
        report_threshold = np.deg2rad(3)  # If deviation > 3 degrees, report event position.
        print(self.events.shape)
        for i in range(self.events.shape[0]):
            event_pos = self.events[i, 8:11]
            # print(event_pos)
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
            if deviation > report_threshold:
                # Compute voxel indices from event_pos.
                # Recall: x_pos = (x - shape[2]/2)*voxel_size, so x = x_pos/voxel_size + shape[2]/2, etc.
                # x_idx = int(round(event_pos[0] / self.voxel_size + self.shape[2] / 2))
                # y_idx = int(round(event_pos[1] / self.voxel_size + self.shape[1] / 2))
                # z_idx = int(round(event_pos[2] / self.voxel_size + self.shape[0] / 2))
                # print(f"Event {i}: voxel indices (z,y,x) = ({z_idx}, {y_idx}, {x_idx}) with deviation = {deviation*180/np.pi:.4f} deg")
                print(f'Event {i}: position = {event_pos}, deviation = {deviation*180/np.pi:.4f} deg')
            
            # print(f"Event {i}: deviation = {deviation*180/np.pi:.4f} deg")
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
        nonuniform_image = gaussian.astype(np.float32)

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


    def test_high_deviation_positions(self):
        """
        Analyze events with collinearity deviation greater than a given threshold (e.g. 3°)
        and print summary statistics of their positions (x, y, z).
        This helps check for systematic bias (e.g., an excess of events at positive z).
        """
        threshold_deg = 3
        threshold_rad = np.deg2rad(threshold_deg)
        high_dev_positions = []  # will store the event physical positions [x, y, z]
        high_dev_devs = []       # will store the deviation values (in radians)

        # Loop over all events.
        for i in range(self.events.shape[0]):
            # event physical position is in columns 8-10.
            event_pos = self.events[i, 8:11]
            det1_pos = self.events[i, 2:5]
            det2_pos = self.events[i, 5:8]
            v1 = det1_pos - event_pos
            v2 = det2_pos - event_pos

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue

            u1 = v1 / norm1
            u2 = v2 / norm2

            dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
            angle = np.arccos(dot)
            deviation = abs(np.pi - angle)
            if deviation > threshold_rad:
                high_dev_positions.append(event_pos)
                high_dev_devs.append(deviation)
        
        high_dev_positions = np.array(high_dev_positions)
        n_high = high_dev_positions.shape[0]
        print(f"\nFound {n_high} events with deviation greater than {threshold_deg}°.")
        if n_high > 0:
            mean_pos = np.mean(high_dev_positions, axis=0)
            std_pos = np.std(high_dev_positions, axis=0)
            min_pos = np.min(high_dev_positions, axis=0)
            max_pos = np.max(high_dev_positions, axis=0)
            print(f"Mean position (x, y, z): {mean_pos}")
            print(f"Std. deviation (x, y, z): {std_pos}")
            print(f"Min position (x, y, z): {min_pos}")
            print(f"Max position (x, y, z): {max_pos}")
            
            # Optionally, report how many events are on positive vs negative z.
            pos_z_count = np.sum(high_dev_positions[:, 2] > 0)
            neg_z_count = np.sum(high_dev_positions[:, 2] <= 0)
            print(f"High deviation events with positive z: {pos_z_count}")
            print(f"High deviation events with non-positive z: {neg_z_count}")
        else:
            print("No high deviation events found.")

if __name__ == '__main__':
    unittest.main()
