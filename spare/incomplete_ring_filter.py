#!/usr/bin/env python3
"""
incomplete_ring_filter.py

This script:
  1) Reads geometry info to figure out how many rings & crystals per ring.
  2) Builds a set of 'missing' detector IDs based on one or more angle ranges
     (e.g. [30,60] deg and [210,245] deg).
  3) Visualizes the missing vs. active detectors in a scatter plot.
  4) For each .lmf listmode file, reads minimal events (det1_id, det2_id),
     drops any event if either detector ID is in missing set,
     and writes a new .lmf file with valid events only.
"""

import os
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt

# --- Configuration ---
CRYSTALS_PER_RING = 544
NUM_RINGS = 72

# The missing angular sectors in degrees (start_deg, end_deg).
MISSING_SECTORS = [(30,60), (210,245)]

# Toggle detailed debug prints
DEBUG = True

def build_missing_detector_ids(crystals_per_ring: int,
                               num_rings: int,
                               missing_sectors) -> set:
    """
    Build a set of detector IDs that lie in the specified missing angle ranges.

    ID is computed as:  detector_id = ring_index * crystals_per_ring + crystal_index
    The angle for crystal_index is: (360 / crystals_per_ring) * crystal_index  (degrees).

    We'll consider a crystal 'missing' if its angle is within any of the specified
    missing angle intervals.
    """
    missing_ids = set()
    if DEBUG:
        print("Debug: Building missing detector IDs...")
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
                # Print some examples
                if DEBUG and len(missing_ids) < 20:
                    print(f"  Debug: ring={ring_idx}, crystal={crystal_idx}, "
                          f"angle={angle_deg:.2f} deg => det_id={det_id} (missing)")

    if DEBUG:
        print(f"Debug: Finished building missing IDs. Total missing = {len(missing_ids)}.")

    return missing_ids

def visualize_detectors(crystals_per_ring: int,
                        num_rings: int,
                        missing_ids: set,
                        missing_sectors):
    """
    Create a scatter plot showing ring index (vertical) vs. angle (horizontal),
    coloring missing detectors in red and active detectors in blue.
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

    plt.figure(figsize=(8,6))
    plt.scatter(angles_active, rings_active, s=3, c='blue', label='Active')
    plt.scatter(angles_missing, rings_missing, s=3, c='red', label='Missing')
    plt.title("Detector Map: Missing vs. Active")
    plt.xlabel("Azimuthal Angle (deg)")
    plt.ylabel("Ring Index")
    for (deg_start, deg_end) in missing_sectors:
        plt.axvspan(deg_start, deg_end, color='red', alpha=0.1)
    plt.xlim(0, 360)
    plt.ylim(0, num_rings)
    plt.legend()
    plt.tight_layout()
    plt.show()

def filter_lmf_file(input_file: str, output_file: str, missing_ids: set):
    """
    Read a minimal listmode .lmf file with dtype=[('det1_id', np.int16), ('det2_id', np.int16)],
    keep only events where neither det1 nor det2 is missing, and write a new .lmf file.
    """
    dtype_minimal = np.dtype([('det1_id', np.int16), ('det2_id', np.int16)])

    if DEBUG:
        print(f"Debug: Reading events from {input_file} ...")

    # Read all events
    events = np.fromfile(input_file, dtype=dtype_minimal)
    n_total = len(events)
    if n_total == 0:
        print(f"  -> {input_file} is empty or no events found.")
        return

    det1_ids = events['det1_id']
    det2_ids = events['det2_id']

    # Build a boolean mask of valid events
    mask_missing_det1 = np.isin(det1_ids, list(missing_ids))
    mask_missing_det2 = np.isin(det2_ids, list(missing_ids))

    # Condition: neither det1 nor det2 is in missing_ids
    valid_mask = ~(mask_missing_det1 | mask_missing_det2)
    filtered_events = events[valid_mask]
    n_filtered = len(filtered_events)

    if n_filtered == 0:
        print(f"  -> All {n_total} events removed, no valid events remain.")
    else:
        # Write them back to .lmf
        filtered_events.tofile(output_file)
        if DEBUG:
            print(f"  Debug: Wrote {n_filtered} events to {output_file}")

    print(f"  -> {n_total} => {n_filtered} events. Saved to {output_file}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmf_folder", type=str, default="listmode_test",
                        help="Folder containing minimal .lmf files to filter.")
    parser.add_argument("--output_folder", type=str, default="listmode_filtered",
                        help="Folder to save the new filtered .lmf files.")
    parser.add_argument("--visualize", action="store_true",
                        help="If set, display a scatter plot of missing vs. active detectors.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # 1) Build the set of missing detectors
    missing_ids = build_missing_detector_ids(
        crystals_per_ring=CRYSTALS_PER_RING,
        num_rings=NUM_RINGS,
        missing_sectors=MISSING_SECTORS
    )
    print(f"Built missing set with {len(missing_ids)} detectors in missing sectors.")

    # 2) (Optional) Visualize the missing vs. active detectors
    if args.visualize:
        visualize_detectors(CRYSTALS_PER_RING, NUM_RINGS, missing_ids, MISSING_SECTORS)

    # 3) Loop through all minimal .lmf files in lmf_folder
    pattern = os.path.join(args.lmf_folder, "listmode_data_minimal_*.lmf")
    lmf_files = sorted(glob.glob(pattern))
    print(f"Found {len(lmf_files)} files matching {pattern}")

    for i, lmf_file in enumerate(lmf_files):
        base_name = os.path.basename(lmf_file)
        out_file = os.path.join(args.output_folder, f"filtered_{base_name}")
        print(f"\n[{i+1}/{len(lmf_files)}] Filtering {lmf_file} => {out_file} ...")
        filter_lmf_file(lmf_file, out_file, missing_ids)

    print("\nDone filtering all files.")

if __name__ == "__main__":
    main()
