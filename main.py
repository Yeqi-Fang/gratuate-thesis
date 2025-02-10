# main.py
import time
import os
import numpy as np
from pet_simulator.geometry import create_pet_geometry
from pet_simulator.simulator import PETSimulator
from pet_simulator.utils import save_events, save_events_binary

# PET scanner configuration information
info = {
    'min_rsector_difference': np.float32(0.0),
    'crystal_length': np.float32(0.0),
    'radius': np.float32(290.56),
    'crystalTransNr': 13,
    'crystalTransSpacing': np.float32(4.03125),
    'crystalAxialNr': 7,
    'crystalAxialSpacing': np.float32(5.36556),
    'submoduleAxialNr': 1,
    'submoduleAxialSpacing': np.float32(0.0),
    'submoduleTransNr': 1,
    'submoduleTransSpacing': np.float32(0.0),
    'moduleTransNr': 1,
    'moduleTransSpacing': np.float32(0.0),
    'moduleAxialNr': 6,
    'moduleAxialSpacing': np.float32(89.82),
    'rsectorTransNr': 32,
    'rsectorAxialNr': 1,
    'TOF': 1,
    'num_tof_bins': np.float32(29.0),
    'tof_range': np.float32(735.7705),
    'tof_fwhm': np.float32(57.71),
    'NrCrystalsPerRing': 416,
    'NrRings': 42,
    'firstCrystalAxis': 0
}

# 179.6391
# 178

def main():
    
    num_events = int(1e9)
    save_events_pos = False
    # Create PET scanner geometry from info
    geometry = create_pet_geometry(info)
    
    # Save the detector lookup table once.
    # (Since LUT depends only on geometry, a dummy image is sufficient.)
    dummy_image = np.ones((1,1,1), dtype=np.float32)
    simulator_for_lut = PETSimulator(geometry, dummy_image, voxel_size=2.78)
    simulator_for_lut.save_detector_positions("detector_lut.txt")
    
    # Define the base directory where the 3D image files are stored.
    base_dir = r"D:\Datasets\dataset\test_npy_crop"
    
    # Create an output directory for listmode data if it doesn't exist.
    output_dir = rf"E:\Datasets\listmode_test\{num_events:d}\cropped"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image file from 3d_image_0.npy to 3d_image_169.npy
    for i in range(36):
        image_filename = f"3d_image_{i}.npy"
        image_path = os.path.join(base_dir, image_filename)
        print(f"\nProcessing {image_filename} ...")
        
        # Load the 3D image (which acts as the probability density distribution)
        image = np.load(image_path)
        print("Image shape:", image.shape)
        
        # Create the simulator with the current image and voxel size
        simulator = PETSimulator(geometry, image, voxel_size=2.78, save_events_pos=save_events_pos)
        
        # Run the simulation
        print("Starting simulation...")
        start_time = time.time()
        events = simulator.simulate_events(num_events=num_events, use_multiprocessing=True)
        end_time = time.time()
        print(f"Generated {len(events)} valid events in {end_time - start_time:.2f} seconds.")
        print("Events shape:", events.shape)
        
        # Save events in binary .lmf format (minimal and full)
        minimal_file = os.path.join(output_dir, f"listmode_data_minimal_{i}_{num_events}")
        save_events_binary(minimal_file, events, save_full_data=False)
        # a = np.random.rand()
        # if a < 0.1:
        #     full_file = os.path.join(output_dir, f"listmode_data_full_{i}_{num_events}.lmf")
        #     save_events_binary(full_file, events, save_full_data=True)
        
        # Optionally, print a sample event.
        if len(events) > 0 and save_events_pos:
            # get detector positions from LUT
            lut = np.loadtxt("detector_lut.txt", skiprows=1, dtype=np.float32)
            det1_pos = lut[int(events[0, 0]), 1:4]
            det2_pos = lut[int(events[0, 1]), 1:4]
            event_pos = events[0, 2:5]
            print("\nSample event (with positions):")
            print("Detector 1 ID:", events[0, 0])
            print("Detector 2 ID:", events[0, 1])
            print("Detector 1 position (x,y,z):", det1_pos)
            print("Detector 2 position (x,y,z):", det2_pos)
            print("Event position (x,y,z):", event_pos)
            # calculate angle 
            # det1_pos = events[0, 2:5]
            # det2_pos = events[0, 5:8]
            
            v1 = det1_pos - event_pos
            v2 = det2_pos - event_pos
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            u1 = v1 / norm1
            u2 = v2 / norm2
            
            dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
            angle = np.arccos(dot)
            
            deviation = abs(np.pi - angle)
            print(f"Event: deviation = {deviation*180/np.pi:.4f} deg")

if __name__ == "__main__":
    main()
