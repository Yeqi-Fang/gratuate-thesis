# PET Reconstruction Pipeline

This project implements a comprehensive pipeline for PET (Positron Emission Tomography) simulation, reconstruction, and analysis, with special emphasis on handling incomplete ring geometries.

## Project Overview

The PET Reconstruction Pipeline provides tools for:

1. **Simulating PET data**: Generate realistic PET listmode data from 3D activity distributions
2. **Reconstructing images**: Convert listmode data to 3D volumes using OSEM algorithm
3. **Creating and analyzing sinograms**: Generate sinogram representations of PET data
4. **Supporting incomplete ring geometries**: Simulate and analyze incomplete PET scanner configurations
5. **Advanced visualization**: Compare complete vs. incomplete ring data with detailed visualizations

## Project Structure

```
pet_reconstruction/
├── data/
│   └── dataset/
│       └── train_npy_crop/         # Input 3D activity distribution images
├── log/                            # Log directories with timestamp subfolders
├── pet_simulator/                  # PET simulation package
│   ├── __init__.py
│   ├── geometry.py                 # PET scanner geometry definitions
│   ├── simulator.py                # Core PET event simulation
│   ├── numba_utils.py              # Accelerated numerical functions
│   └── utils.py                    # Utilities for saving/loading data
├── main.py                         # Main simulation & reconstruction pipeline
├── generate_reconstruct.py         # Enhanced reconstruction with visualizations
├── reconstruction_all.py           # Batch reconstruction from listmode data
├── listmode_to_incomplete.py       # Convert complete data to incomplete ring data
├── enhanced_visualization.py       # Advanced visualization utilities
├── outlier_detection.py            # Functions for outlier detection and removal
└── README.md                       # This file
```

## Output Directory Structure

```
sinogram/
└── reconstruction_npy_full_train/
    ├── <num_events>/               # Results organized by event count
    │   ├── listmode/               # Listmode data (.npz files)
    │   ├── sinogram/               # Complete sinograms (.npy files)
    │   ├── incomplete/             # Incomplete ring data
    │   │   ├── listmode_incomplete/    # Filtered listmode data
    │   │   ├── sinogram_incomplete/    # Incomplete sinograms
    │   │   └── log_incomplete/         # Logs and visualizations
    │   └── reconstructed*.npy      # Reconstructed 3D volumes
    └── ...
```

## Installation and Dependencies

### Requirements

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib
- Numba (for acceleration)
- pytomography (for reconstruction)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pet-reconstruction
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy matplotlib numba
   # Install pytomography according to its documentation
   ```

## Usage Examples

### 1. Full Simulation and Reconstruction Pipeline

Run the main simulation and reconstruction pipeline:

```bash
python main.py
```

This will:
- Load 3D activity distributions from `data/dataset/train_npy_crop/`
- Simulate PET events using a defined scanner geometry
- Save listmode data and generate sinograms
- Reconstruct 3D volumes using OSEM
- Create visualizations in the log directory

### 2. Convert Complete Listmode Data to Incomplete Ring Data

Generate incomplete ring data by removing detector sectors:

```bash
python listmode_to_incomplete.py \
    --input_dir /mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000/listmode \
    --output_dir /mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000/incomplete \
    --sinogram_dir /mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000/sinogram \
    --num_events 2000000000 \
    --visualize \
    --missing_start1 30 --missing_end1 90 \
    --missing_start2 210 --missing_end2 270
```

This simulates a PET scanner with two 60-degree missing sectors and generates comparison visualizations.

### 3. Batch Reconstruction

Reconstruct multiple listmode files:

```bash
python reconstruction_all.py \
    --lmf_root /mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000/listmode \
    --lut_file detector_lut.txt \
    --output_dir /mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000 \
    --outlier False
```

## Key Components

### PET Simulator

The `pet_simulator` package contains:
- Geometric definitions of the PET scanner
- Monte Carlo simulation for positron annihilation events
- Ray-tracing to detect coincident events
- Efficient event generation using Numba-accelerated functions

### Reconstruction

Reconstruction is handled by:
- `PETLMSystemMatrix` for system modeling
- OSEM algorithm for iterative reconstruction
- Optional PSF correction and outlier removal

### Visualization

The project includes enhanced visualization capabilities:
- Multi-slice sinogram visualizations
- Complete vs. incomplete ring comparisons
- 3D volume visualizations from multiple perspectives
- Side-by-side comparison of original and reconstructed images

### Incomplete Ring Simulation

The incomplete ring functionality:
- Simulates missing detector sectors by angular position
- Filters listmode data to remove events involving missing detectors
- Generates comparison visualizations to show the impact of missing data
- Provides tools to analyze the effects on reconstruction quality

## Performance Optimizations

- Multithreaded processing for I/O operations
- Numba acceleration for computationally intensive functions
- Background processing for file saving and visualization generation
- Memory-efficient data handling for large datasets

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your changes

