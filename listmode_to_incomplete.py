#!/usr/bin/env python3
"""
listmode_to_incomplete.py

This script:
  1) Reads complete listmode data (.npz files)
  2) Filters out events involving detectors in specified angular ranges (creating "incomplete ring" data)
  3) Generates sinograms from the filtered data
  4) Creates comparison visualizations between complete and incomplete sinograms
  5) Saves both the incomplete listmode data and sinograms

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
import threading
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid GUI errors
import matplotlib.pyplot as plt
from datetime import datetime

from pytomography.io.PET import gate

# Import our enhanced visualization functions
from enhanced_visualization import (
    visualize_sinogram_multislice,
    visualize_sinogram_multi_perspective,
    compare_sinograms
)

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

def load_complete_sinogram(sinogram_dir, index, num_events):
    """
    Load the complete sinogram data from the given directory.
    
    Args:
        sinogram_dir: Base directory containing sinogram data
        index: Index number of the data
        num_events: Number of events
    
    Returns:
        Complete sinogram data if found, None otherwise
    """
    # 构造完整环正弦图文件路径
    complete_path = os.path.join(
        sinogram_dir, 
        f"reconstructed_index{index}_num{num_events}.npy"
    )
    
    # 尝试加载完整环正弦图
    if os.path.exists(complete_path):
        try:
            return np.load(complete_path)
        except Exception as e:
            print(f"Error loading complete sinogram from {complete_path}: {e}")
    
    return None

def process_and_compare_sinograms_background(complete_sinogram, incomplete_sinogram, output_dir, log_dir, image_index):
    """
    Process and compare complete and incomplete sinograms in a background thread.
    
    Args:
        complete_sinogram: Complete ring sinogram data
        incomplete_sinogram: Incomplete ring sinogram data
        output_dir: Output directory for saving results
        log_dir: Log directory for visualizations
        image_index: Index of the image being processed
    """
    def _process_and_compare():
        # 确保在线程中使用非交互式后端
        matplotlib.use('Agg')
        
        try:
            # 创建可视化目录
            vis_dir = os.path.join(log_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 生成比较可视化
            compare_path = os.path.join(
                vis_dir, 
                f"sinogram_comparison_index{image_index}.png"
            )
            
            # 使用增强的可视化函数生成比较图
            compare_sinograms(
                complete_sinogram=complete_sinogram,
                incomplete_sinogram=incomplete_sinogram,
                output_path=compare_path,
                title=f"Complete vs Incomplete Sinogram Comparison (Index {image_index})",
                num_slices=3
            )
            
            print(f"  -> Saved sinogram comparison to {compare_path}")
            
            # 可选：为不完整环正弦图单独创建多切片可视化
            multislice_path = os.path.join(
                vis_dir, 
                f"incomplete_sinogram_multislice_index{image_index}.png"
            )
            
            visualize_sinogram_multislice(
                sinogram=incomplete_sinogram,
                output_path=multislice_path,
                title=f"Incomplete Ring Sinogram Multislice (Index {image_index})",
                num_slices=8
            )
            
            print(f"  -> Saved incomplete sinogram multislice to {multislice_path}")
            
        except Exception as e:
            print(f"Warning: Failed to generate sinogram comparison: {e}")
            import traceback
            traceback.print_exc()
    
    thread = threading.Thread(target=_process_and_compare)
    thread.daemon = False  # 确保线程不会随主程序退出而终止
    thread.start()
    return thread

def process_listmode_file(input_file, output_dir, complete_sinogram_dir, log_dir, missing_ids, num_events, vis_level=2):
    """
    Process a single listmode file: filter it and generate a sinogram.
    
    Args:
        input_file: Path to the input listmode .npz file
        output_dir: Base output directory
        complete_sinogram_dir: Directory containing complete sinograms for comparison
        log_dir: Directory for visualizations and logs
        missing_ids: Set of detector IDs considered missing
        num_events: Event count for output path construction
        vis_level: Visualization detail level (0=minimal, 1=basic, 2=detailed)
    """
    # Extract index from filename using regex
    filename = os.path.basename(input_file)
    match = re.search(r'reconstructed_index(\d+)_num|listmode_data_minimal_(\d+)_', filename)
    if not match:
        print(f"Error: Could not parse index from filename: {filename}")
        return
    
    # 提取索引，考虑不同的文件命名模式
    index = match.group(1) if match.group(1) else match.group(2)
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
    
    # Save filtered listmode data in background
    out_lm_path = os.path.join(incomplete_lm_dir, f"incomplete_index{index}_num{num_events}.npz")
    save_thread = threading.Thread(
        target=lambda: np.savez_compressed(out_lm_path, listmode=filtered_events)
    )
    save_thread.daemon = False
    save_thread.start()
    
    # Convert filtered events to detector IDs for sinogram generation
    detector_ids = torch.from_numpy(filtered_events).to(torch.int32)
    
    # Generate sinogram
    print("Generating sinogram from incomplete data...")
    incomplete_sinogram = gate.listmode_to_sinogram(detector_ids, info)
    
    # Save incomplete sinogram
    out_sinogram_path = os.path.join(incomplete_sinogram_dir, f"incomplete_index{index}_num{num_events}")
    np.save(out_sinogram_path, incomplete_sinogram.numpy().astype(np.float32))
    print(f"Saved incomplete sinogram to {out_sinogram_path}")
    
    # 尝试加载对应的完整环正弦图进行比较
    complete_sinogram = load_complete_sinogram(complete_sinogram_dir, index, num_events)
    
    # 如果找到完整环正弦图，在后台线程中进行比较可视化
    if complete_sinogram is not None and vis_level >= 1:
        print("Found matching complete sinogram, generating comparison...")
        comparison_thread = process_and_compare_sinograms_background(
            complete_sinogram=complete_sinogram,
            incomplete_sinogram=incomplete_sinogram.numpy(),
            output_dir=output_dir,
            log_dir=log_dir,
            image_index=index
        )
    else:
        print("No matching complete sinogram found for comparison.")
        # 即使没有完整环数据，也为不完整环数据单独创建可视化
        if vis_level >= 1:
            visualize_sinogram_multislice(
                sinogram=incomplete_sinogram.numpy(),
                output_path=os.path.join(log_dir, f"incomplete_sinogram_index{index}.png"),
                title=f"Incomplete Ring Sinogram (Index {index})",
                num_slices=6
            )
    
    # 等待后台保存线程完成
    save_thread.join()
    print(f"Completed processing index {index} in {time.time() - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Convert complete listmode data to incomplete ring data')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing complete listmode .npz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory')
    parser.add_argument('--sinogram_dir', type=str, 
                        help='Directory containing complete sinograms for comparison')
    parser.add_argument('--num_events', type=int, required=True,
                        help='Number of events (used for path construction)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate detector coverage visualization')
    parser.add_argument('--vis_level', type=int, default=2, choices=[0, 1, 2],
                        help='Visualization detail level (0=minimal, 1=basic, 2=detailed)')
    parser.add_argument('--missing_start1', type=float, default=30,
                        help='Start angle of first missing sector (degrees)')
    parser.add_argument('--missing_end1', type=float, default=90,
                        help='End angle of first missing sector (degrees)')
    parser.add_argument('--missing_start2', type=float, default=210,
                        help='Start angle of second missing sector (degrees)')
    parser.add_argument('--missing_end2', type=float, default=270,
                        help='End angle of second missing sector (degrees)')
    args = parser.parse_args()
    
    # 根据命令行参数设置缺失扇区
    missing_sectors = [
        (args.missing_start1, args.missing_end1),
        (args.missing_start2, args.missing_end2)
    ]
    
    # Ensure base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create log directory
    log_dir = os.path.join(args.output_dir, 'log_incomplete', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")
    
    # If sinogram_dir not provided, use default location
    sinogram_dir = args.sinogram_dir
    if not sinogram_dir:
        # 尝试构造默认位置：与输入目录平级的sinogram目录
        parent_dir = os.path.dirname(args.input_dir)
        sinogram_dir = os.path.join(parent_dir, 'sinogram')
        if not os.path.exists(sinogram_dir):
            print(f"Complete sinogram directory not found at {sinogram_dir}")
            print("Visualizations will not include comparisons")
            sinogram_dir = None
    
    # Build missing detector IDs set
    missing_ids = build_missing_detector_ids(
        crystals_per_ring=info['NrCrystalsPerRing'],
        num_rings=info['NrRings'],
        missing_sectors=missing_sectors
    )
    
    # Generate detector coverage visualization
    if args.visualize:
        visualize_detector_coverage(
            crystals_per_ring=info['NrCrystalsPerRing'],
            num_rings=info['NrRings'],
            missing_ids=missing_ids,
            missing_sectors=missing_sectors,
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
    
    # 跟踪所有后台线程
    all_threads = []
    
    # Process each file
    for i, lm_file in enumerate(listmode_files):
        print(f"\n[{i+1}/{len(listmode_files)}] Processing {lm_file}")
        
        # 处理单个文件
        process_listmode_file(
            input_file=lm_file,
            output_dir=args.output_dir,
            complete_sinogram_dir=sinogram_dir,
            log_dir=log_dir,
            missing_ids=missing_ids,
            num_events=args.num_events,
            vis_level=args.vis_level
        )
    
    # 等待所有后台线程完成
    for thread in all_threads:
        thread.join()
    
    print("\nAll files processed. Incomplete ring data generation complete.")

if __name__ == "__main__":
    main()