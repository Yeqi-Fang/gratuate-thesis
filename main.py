# main.py
import time
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
    
    num_events = int(1e6)
    # Load the 3D image that acts as the probability density distribution.
    image = np.load('3d_image_2.npy')
    
    # Create PET scanner geometry from info
    geometry = create_pet_geometry(info)
    
    # Instantiate the simulator with the image and voxel size
    simulator = PETSimulator(geometry, image, voxel_size=2.78)
    
    # Save the detector lookup table
    simulator.save_detector_positions("detector_lut.txt")
    
    # Run the simulation
    print("Starting simulation...")
    start_time = time.time()
    events = simulator.simulate_events(num_events=num_events)
    end_time = time.time()
    
    # Save events in both minimal and full formats
    save_events(f"listmode/listmode_data_minimal_{num_events}.txt", events, save_full_data=False)
    save_events(f"listmode/listmode_data_full_{num_events}.txt", events, save_full_data=True)
    
    # Save events in binary .lmf format (minimal and full)
    save_events_binary("listmode/listmode_data_minimal.lmf", events, save_full_data=False)
    save_events_binary("listmode/listmode_data_full.lmf", events, save_full_data=True)
    
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
