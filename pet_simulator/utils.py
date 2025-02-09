# pet_simulator/utils.py
import numpy as np
import h5py

def save_detector_lut(filename: str, detector_positions: np.ndarray):
    """
    Save detector lookup table (LUT).
    The table includes detector IDs and positions (x, y, z).
    """
    detector_ids = np.arange(len(detector_positions))
    lut_data = np.column_stack((detector_ids, detector_positions))
    header = "detector_id x y z"
    fmt = ['%d'] + ['%.6f'] * 3
    np.savetxt(filename, lut_data, fmt=fmt, header=header)

def save_events(filename: str, events: np.ndarray, save_full_data: bool = False):
    """
    Save events data to a text file.
    If save_full_data is True, save both detector IDs and positions;
    otherwise, save only the detector IDs.
    """
    if save_full_data:
        header = ("det1_id det2_id "
                  "det1_x det1_y det1_z "
                  "det2_x det2_y det2_z "
                  "event_x event_y event_z")
        fmt = ['%d', '%d'] + ['%.6f'] * 9
        np.savetxt(filename, events, fmt=fmt, header=header)
    else:
        header = "det1_id det2_id"
        detector_ids = events[:, :2]
        np.savetxt(filename, detector_ids, fmt='%d', header=header)

def save_events_binary(filename: str, events: np.ndarray, save_full_data: bool = False):
    """
    Save events data in binary .lmf format.
    The binary file can be read using np.fromfile with a structured dtype.
    
    For minimal data: only det1_id and det2_id (stored as int16).
    For full data: also include detector and event positions (stored as float32).
    """
    if save_full_data:
        dtype_full = np.dtype([
            ('det1_id', np.uint16), ('det2_id', np.uint16),
            ('event_x', np.float32), ('event_y', np.float32), ('event_z', np.float32)
        ])
        structured_array = np.empty(events.shape[0], dtype=dtype_full)
        structured_array['det1_id'] = events[:, 0].astype(np.uint16)
        structured_array['det2_id'] = events[:, 1].astype(np.uint16)
        structured_array['event_x'] = events[:, 2].astype(np.float32)
        structured_array['event_y'] = events[:, 3].astype(np.float32)
        structured_array['event_z'] = events[:, 4].astype(np.float32)
        
        # lut = np.loadtxt("detector_lut.txt", skiprows=1, dtype=np.float32)
        # det1_pos = lut[int(events[0, 0]), 1:4]
        # det2_pos = lut[int(events[0, 1]), 1:4]
        # structured_array['det1_x'] = det1_pos[0].astype(np.float32)
        # structured_array['det1_y'] = det1_pos[1].astype(np.float32)
        # structured_array['det1_z'] = det1_pos[2].astype(np.float32)
        # structured_array['det2_x'] = det2_pos[0].astype(np.float32)
        # structured_array['det2_y'] = det2_pos[1].astype(np.float32)
        # structured_array['det2_z'] = det2_pos[2].astype(np.float32)
    else:
        dtype_minimal = np.dtype([
            ('det1_id', np.uint16), ('det2_id', np.uint16)
        ])
        structured_array = np.empty(events.shape[0], dtype=dtype_minimal)
        structured_array['det1_id'] = events[:, 0].astype(np.uint16)
        structured_array['det2_id'] = events[:, 1].astype(np.uint16)
        
    # with h5py.File(f"{filename}.h5", 'w') as f:
    #     # 创建数据集时启用gzip压缩，调整chunks参数优化压缩率
    #     dset = f.create_dataset('listmode', data=structured_array, chunks=True, compression='gzip', compression_opts=9)
    np.savez_compressed(f"{filename}.npz", listmode=structured_array)
    # structured_array.tofile(f"{filename}.lmf")
