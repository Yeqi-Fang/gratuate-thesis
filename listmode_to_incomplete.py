#!/usr/bin/env python3
"""
listmode_to_incomplete.py

This script:
  1) Reads complete listmode data (.npz files)
  2) Filters out events involving detectors in specified angular ranges (creating "incomplete ring" data)
  3) Generates sinograms from the filtered data
  4) Saves both the incomplete listmode data and sinograms

Usage:
  python listmode_to_incomplete.py --input_dir /path/to/complete/listmode --output_dir /path/to/output --num_events 2000000000
"""

import os
import re
import glob
import numpy as np
import torch
import time
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid GUI errors
import matplotlib.pyplot as plt
from datetime import datetime

from pytomography.io.PET import gate

# PET scanner configuration (from main.py)
info = {
    'min_rsector_difference': np.float32(0.0),
    'crystal_length': np.float32(0.0),
    'radius': np.float32(253.71),
    'crystalTransNr': 13,
    'crystalTransSpacing': np.float32(4.01648),
    'crystalAxialNr': 7,
    'crystalAxialSpacing': np.float32(5.36556),
    'submoduleAxialNr': 1,
    'submoduleAxialSpacing': np.float32(0.0),
    'submoduleTransNr': 1,
    'submoduleTransSpacing': np.float32(0.0),
    'moduleTransNr': 1,
    'moduleTransSpacing': np.float32(0.0),
    'moduleAxialNr': 6,
    'moduleAxialSpacing': np.float32(37.55892),
    'rsectorTransNr': 28,
    'rsectorAxialNr': 1,
    'TOF': 0,
    'NrCrystalsPerRing': 364,  # 13 * 7 * 4
    'NrRings': 42,
    'firstCrystalAxis': 0
}

# Define missing angular sectors in degrees (start_deg, end_deg)
# Adjust these values to control how much of the ring is missing
MISSING_SECTORS = [(30, 90), (210, 270)]  # Example: 60-degree gaps at opposite sides
DEBUG = True

def build_missing_detector_ids(crystals_per_ring, num_rings, missing_sectors):
    """
    Build a set of detector IDs that lie in the specified missing angle ranges.

    ID is computed as: detector_id = ring_index * crystals_per_ring + crystal_index
    The angle for crystal_index is: (360 / crystals_per_ring) * crystal_index (degrees).

    Args:
        crystals_per_ring: Number of crystals in each ring
        num_rings: Number of axial rings
        missing_sectors: List of (start_angle, end_angle) tuples in degrees

    Returns:
        Set of detector IDs to be considered "missing"
    """
    missing_ids = set()
    if DEBUG:
        print("Building missing detector IDs...")
        print(f"  crystals_per_ring = {crystals_per_ring}, num_rings = {num_rings}")
        print(f"  missing_sectors = {missing_sectors}")

    for ring_idx in range(num_rings):
        for crystal_idx in range(crystals_per_ring):
            angle_deg = (360.0 / crystals_per_ring) * crystal_idx
            # Check if angle_deg is in a missing sector
            is_missing = False
            for (deg_start, deg_end) in missing_sectors:
                if deg_start <= angle_deg <= deg_end:
                    is_missing = True
                    break

            if is_missing:
                det_id = ring_idx * crystals_per_ring + crystal_idx
                missing_ids.add(det_id)
                # Print some examples for debugging
                if DEBUG and len(missing_ids) < 10:
                    print(f"  Debug: ring={ring_idx}, crystal={crystal_idx}, "
                          f"angle={angle_deg:.2f} deg => det_id={det_id} (missing)")

    if DEBUG:
        print(f"Built missing detector set with {len(missing_ids)} detectors "
              f"({(len(missing_ids) / (crystals_per_ring * num_rings) * 100):.1f}% of total).")

    return missing_ids

def visualize_detector_coverage(crystals_per_ring, num_rings, missing_ids, missing_sectors, output_dir):
    """
    Create a visualization showing the missing detector coverage.
    
    Args:
        crystals_per_ring: Number of crystals in each ring
        num_rings: Number of axial rings
        missing_ids: Set of detector IDs considered missing
        missing_sectors: List of (start_angle, end_angle) tuples defining missing sectors
        output_dir: Directory to save the visualization
    """
    angles_active = []
    rings_active = []
    angles_missing = []
    rings_missing = []

    for ring_idx in range(num_rings):
        for crystal_idx in range(crystals_per_ring):
            angle_deg = (360.0 / crystals_per_ring) * crystal_idx
            det_id = ring_idx * crystals_per_ring + crystal_idx
            if det_id in missing_ids:
                angles_missing.append(angle_deg)
                rings_missing.append(ring_idx)
            else:
                angles_active.append(angle_deg)
                rings_active.append(ring_idx)

    plt.figure(figsize=(10, 6))
    plt.scatter(angles_active, rings_active, s=2, c='blue', label='Active Detectors')
    plt.scatter(angles_missing, rings_missing, s=2, c='red', label='Missing Detectors')
    plt.title("Detector Coverage: Complete vs. Incomplete Ring")
    plt.xlabel("Azimuthal Angle (degrees)")
    plt.ylabel("Ring Index")
    
    # Highlight missing sectors
    for (deg_start, deg_end) in missing_sectors:
        plt.axvspan(deg_start, deg_end, color='red', alpha=0.1)
    
    plt.xlim(0, 360)
    plt.ylim(0, num_rings)
    plt.legend()
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "detector_coverage.png"), dpi=300)
    plt.close()
    print(f"Saved detector coverage visualization to {output_dir}/detector_coverage.png")

def filter_listmode_data(events, missing_ids):
    """
    Filter listmode data to remove events involving missing detectors.
    
    Args:
        events: Numpy array of detector ID pairs
        missing_ids: Set of detector IDs considered missing
    
    Returns:
        Filtered events array
    """
    # Extract detector IDs
    det1_ids = events[:, 0]
    det2_ids = events[:, 1]
    
    # Build boolean mask for valid events (neither detector in missing_ids)
    mask_missing_det1 = np.isin(det1_ids, list(missing_ids))
    mask_missing_det2 = np.isin(det2_ids, list(missing_ids))
    
    # Keep only events where neither detector is missing
    valid_mask = ~(mask_missing_det1 | mask_missing_det2)
    filtered_events = events[valid_mask]
    
    return filtered_events

def process_listmode_file(input_file, output_dir, log_dir, missing_ids, num_events):
    """
    Process a single listmode file: filter it and generate a sinogram.
    
    Args:
        input_file: Path to the input listmode .npz file
        output_dir: Base output directory
        log_dir: Directory for visualizations and logs
        missing_ids: Set of detector IDs considered missing
        num_events: Event count for output path construction
    """
    # Extract index from filename using regex
    filename = os.path.basename(input_file)
    match = re.search(r'reconstructed_index(\d+)_num', filename)
    if not match:
        print(f"Error: Could not parse index from filename: {filename}")
        return
    
    index = match.group(1)
    print(f"\nProcessing listmode data for index {index}...")
    
    start_time = time.time()
    
    # Load listmode data
    try:
        events = np.load(input_file)
        if isinstance(events, np.ndarray):
            events_data = events
        else:
            # It's an .npz file
            if 'listmode' in events:
                events_data = events['listmode']
            else:
                events_data = next(iter(events.values()))
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        return
    
    print(f"Loaded {len(events_data)} events from {input_file}")
    
    # Filter events to create incomplete ring data
    filtered_events = filter_listmode_data(events_data, missing_ids)
    print(f"Filtered to {len(filtered_events)} events ({len(filtered_events)/len(events_data)*100:.1f}% of original)")
    
    if len(filtered_events) == 0:
        print("Warning: All events were filtered out. Check missing sectors configuration.")
        return
    
    # Set up output directories
    incomplete_lm_dir = os.path.join(output_dir, 'listmode_incomplete')
    incomplete_sinogram_dir = os.path.join(output_dir, 'sinogram_incomplete')
    
    os.makedirs(incomplete_lm_dir, exist_ok=True)
    os.makedirs(incomplete_sinogram_dir, exist_ok=True)
    
    # Save filtered listmode data
    out_lm_path = os.path.join(incomplete_lm_dir, f"incomplete_index{index}_num{num_events}.npz")
    np.savez_compressed(out_lm_path, listmode=filtered_events)
    print(f"Saved incomplete listmode data to {out_lm_path}")
    
    # Convert to tensor for sinogram generation
    detector_ids = torch.from_numpy(filtered_events).to(torch.int32)
    
    # Generate sinogram
    print("Generating sinogram from incomplete data...")
    sinogram = gate.listmode_to_sinogram(detector_ids, info)
    
    # Save sinogram
    out_sinogram_path = os.path.join(incomplete_sinogram_dir, f"incomplete_index{index}_num{num_events}")
    np.save(out_sinogram_path, sinogram.numpy().astype(np.float32))
    print(f"Saved incomplete sinogram to {out_sinogram_path}")
    
    # Visualize sinogram
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        im = ax.imshow(sinogram.numpy()[0, :, :42], cmap='magma')
        ax.set_title(f'Incomplete Ring Sinogram (Index {index})')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        vis_filename = os.path.join(log_dir, f"incomplete_sinogram_{index}.pdf")
        plt.savefig(vis_filename, dpi=300)
        plt.close(fig)
        print(f"Saved sinogram visualization to {vis_filename}")
    except Exception as e:
        print(f"Warning: Failed to visualize sinogram: {e}")
    
    total_time = time.time() - start_time
    print(f"Completed processing index {index} in {total_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Convert complete listmode data to incomplete ring data')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing complete listmode .npz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory')
    parser.add_argument('--num_events', type=int, required=True,
                        help='Number of events (used for path construction)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate detector coverage visualization')
    args = parser.parse_args()
    
    # Ensure base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create log directory
    log_dir = os.path.join(args.output_dir, 'log_incomplete', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")
    
    # Build missing detector IDs set
    missing_ids = build_missing_detector_ids(
        crystals_per_ring=info['NrCrystalsPerRing'],
        num_rings=info['NrRings'],
        missing_sectors=MISSING_SECTORS
    )
    
    # Generate detector coverage visualization
    if args.visualize:
        visualize_detector_coverage(
            crystals_per_ring=info['NrCrystalsPerRing'],
            num_rings=info['NrRings'],
            missing_ids=missing_ids,
            missing_sectors=MISSING_SECTORS,
            output_dir=log_dir
        )
    
    # Find all listmode files
    if os.path.isdir(args.input_dir):
        # Look for .npz files in the directory
        listmode_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    else:
        # Treat input_dir as a single file
        listmode_files = [args.input_dir] if os.path.exists(args.input_dir) else []
    
    print(f"Found {len(listmode_files)} listmode files to process")
    
    if not listmode_files:
        print("No input files found. Check the --input_dir path.")
        return
    
    # Process each file
    for i, lm_file in enumerate(listmode_files):
        print(f"\n[{i+1}/{len(listmode_files)}] Processing {lm_file}")
        process_listmode_file(
            input_file=lm_file,
            output_dir=args.output_dir,
            log_dir=log_dir,
            missing_ids=missing_ids,
            num_events=args.num_events
        )
    
    print("\nAll files processed. Incomplete ring data generation complete.")

if __name__ == "__main__":
    main()