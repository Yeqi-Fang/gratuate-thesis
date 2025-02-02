# main.py
import time
import os
import numpy as np
from pet_simulator.geometry import create_pet_geometry
from pet_simulator.simulator import PETSimulator
from pet_simulator.utils import save_events, save_events_binary

# PET scanner configuration information
info = {
    'min_rsector_difference': np.float64(0.0),
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
    'firstCrystalAxis': 0
}

def main():
    
    num_events = int(1e8)
    
    # Create PET scanner geometry from info
    geometry = create_pet_geometry(info)
    
    # Save the detector lookup table once.
    # (Since LUT depends only on geometry, a dummy image is sufficient.)
    dummy_image = np.ones((1,1,1), dtype=np.float64)
    simulator_for_lut = PETSimulator(geometry, dummy_image, voxel_size=2.78)
    simulator_for_lut.save_detector_positions("detector_lut.txt")
    
    # Define the base directory where the 3D image files are stored.
    base_dir = r"D:\Datasets\dataset\test_npy"
    
    # Create an output directory for listmode data if it doesn't exist.
    output_dir = "listmode_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image file from 3d_image_0.npy to 3d_image_169.npy
    for i in range(70):
        image_filename = f"3d_image_{i}.npy"
        image_path = os.path.join(base_dir, image_filename)
        print(f"\nProcessing {image_filename} ...")
        
        # Load the 3D image (which acts as the probability density distribution)
        image = np.load(image_path)
        
        # Create the simulator with the current image and voxel size
        simulator = PETSimulator(geometry, image, voxel_size=2.78)
        
        # Run the simulation
        print("Starting simulation...")
        start_time = time.time()
        events = simulator.simulate_events(num_events=num_events)
        end_time = time.time()
        print(f"Generated {len(events)} valid events in {end_time - start_time:.2f} seconds.")
        print("Events shape:", events.shape)
        
        # Save events in binary .lmf format (minimal and full)
        minimal_file = os.path.join(output_dir, f"listmode_data_minimal_{i}_{num_events}.lmf")
        save_events_binary(minimal_file, events, save_full_data=False)
        a = np.random.rand()
        if a < 0.1:
            full_file = os.path.join(output_dir, f"listmode_data_full_{i}_{num_events}.lmf")
            save_events_binary(full_file, events, save_full_data=True)
        
        # Optionally, print a sample event.
        if len(events) > 0:
            print("\nSample event (with positions):")
            print("Detector 1 ID:", events[0, 0])
            print("Detector 2 ID:", events[0, 1])
            print("Detector 1 position (x,y,z):", events[0, 2:5])
            print("Detector 2 position (x,y,z):", events[0, 5:8])
            print("Event position (x,y,z):", events[0, 8:11])


if __name__ == "__main__":
    main()
