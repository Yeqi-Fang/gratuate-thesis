#!/usr/bin/env python3
"""
incomplete_ring_filter.py

This script:
  1) Reads geometry info to figure out how many rings & crystals per ring.
  2) Builds a set of 'missing' detector IDs based on one or more angle ranges
     (e.g. [30,60] deg and [210,245] deg).
  3) For each .lmf listmode file, reads minimal events (det1_id, det2_id),
     drops any event if either detector ID is in missing set,
     and writes a new .lmf file with valid events only.
"""

import os
import glob
import numpy as np

# Example geometry (matching your config)
CRYSTALS_PER_RING = 544
NUM_RINGS = 72

# The missing angular sectors in degrees (start_deg, end_deg). 
# For example: [30,60] and [210,245].
MISSING_SECTORS = [(30,60), (210,245)]

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
    for ring_idx in range(num_rings):
        # ring does not affect angle in x-y plane
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
    return missing_ids

def filter_lmf_file(input_file: str, output_file: str, missing_ids: set):
    """
    Read a minimal listmode .lmf file with dtype=[('det1_id', np.int16), ('det2_id', np.int16)],
    keep only events where neither det1 nor det2 is missing, and write a new .lmf file.
    """
    dtype_minimal = np.dtype([('det1_id', np.int16), ('det2_id', np.int16)])
    
    # Read all events
    events = np.fromfile(input_file, dtype=dtype_minimal)
    if len(events) == 0:
        print(f"  -> {input_file} is empty or no events found.")
        return
    
    # Build a boolean mask of valid events
    det1_ids = events['det1_id']
    det2_ids = events['det2_id']
    
    # Condition: neither det1 nor det2 is in missing_ids
    valid_mask = ~(np.isin(det1_ids, list(missing_ids)) | np.isin(det2_ids, list(missing_ids)))
    filtered_events = events[valid_mask]
    
    if len(filtered_events) == 0:
        print(f"  -> All events removed, no valid events remain.")
    else:
        # Write them back to .lmf
        filtered_events.tofile(output_file)
    print(f"  -> {len(events)} => {len(filtered_events)} events. Saved to {output_file}.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmf_folder", type=str, default="listmode_test",
                        help="Folder containing minimal .lmf files to filter.")
    parser.add_argument("--output_folder", type=str, default="listmode_filtered",
                        help="Folder to save the new filtered .lmf files.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    
    # 1) Build the set of missing detectors
    missing_ids = build_missing_detector_ids(
        crystals_per_ring=CRYSTALS_PER_RING,
        num_rings=NUM_RINGS,
        missing_sectors=MISSING_SECTORS
    )
    print(f"Built missing set with {len(missing_ids)} detectors in missing sectors.")
    
    # 2) Loop through all minimal .lmf files in lmf_folder
    pattern = os.path.join(args.lmf_folder, "listmode_data_minimal_*.lmf")
    lmf_files = sorted(glob.glob(pattern))
    print(f"Found {len(lmf_files)} files matching {pattern}")
    
    for lmf_file in lmf_files:
        base_name = os.path.basename(lmf_file)
        out_file = os.path.join(args.output_folder, f"filtered_{base_name}")
        print(f"Filtering {lmf_file} => {out_file} ...")
        filter_lmf_file(lmf_file, out_file, missing_ids)

    print("Done filtering all files.")

if __name__ == "__main__":
    main()
