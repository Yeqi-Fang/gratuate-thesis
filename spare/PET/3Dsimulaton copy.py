#!/usr/bin/env python

import numpy as np
import math, random
from tqdm import tqdm

##########################
# 1) PET geometry from 'info'
##########################
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

##########################
# 2) Object meta
##########################
class ObjectMeta:
    def __init__(self, dr, shape):
        # dr is the voxel size, e.g., (2.78, 2.78, 2.78) in mm
        # shape is the 3D dimension, e.g., (128,128,128)
        self.dr = dr
        self.shape = shape

object_meta = ObjectMeta(dr=(2.78, 2.78, 2.78), shape=(128,128,128))

##########################
# Utility: random direction
##########################
def random_unit_3d():
    """
    Return a random 3D unit vector, uniformly distributed on the unit sphere.
    """
    z = random.uniform(-1, 1)
    phi = random.uniform(0, 2*math.pi)
    r_xy = math.sqrt(max(0.0, 1 - z*z))
    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)
    return np.array([x, y, z], dtype=np.float32)

##########################
# 3) Build crystal positions
##########################
def build_detector_positions(info):
    """
    Build an array of 3D coordinates (x, y, z) for the center of each crystal,
    plus an array of crystal IDs (0..N-1).
    """
    radius = float(info['radius'])  
    nr_crystals_per_ring = int(info['NrCrystalsPerRing'])  # e.g. 288
    nr_rings = int(info['NrRings'])                       # e.g. 54
    ring_gap = float(info['crystalAxialSpacing'])         # e.g. 6.6533

    positions = []
    crystal_ids = []
    for ring_i in range(nr_rings):
        # center z for ring_i
        z_i = (ring_i - (nr_rings - 1)/2.0) * ring_gap
        for c_i in range(nr_crystals_per_ring):
            angle = 2.0 * math.pi * c_i / nr_crystals_per_ring
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions.append([x, y, z_i])
            crystal_ids.append(ring_i*nr_crystals_per_ring + c_i)
    return np.array(positions, dtype=np.float32), np.array(crystal_ids, dtype=np.int32)

##########################
# 4) Intersection logic
##########################
def find_crystal_id(x0, y0, z0, direction, info):
    """
    Return crystal ID if the photon hits the cylindrical ring, else -1.
    This simplified approach:
      1) Solve for intersection with cylinder radius = info['radius']
      2) Check if z is in [z_min, z_max]
      3) Convert z -> ring index, angle -> crystal index
    """
    R = float(info['radius'])
    nr_crystals_per_ring = int(info['NrCrystalsPerRing'])
    nr_rings = int(info['NrRings'])
    ring_gap = float(info['crystalAxialSpacing'])

    dx, dy, dz = direction
    A = dx*dx + dy*dy
    B = 2*(x0*dx + y0*dy)
    C = x0*x0 + y0*y0 - R*R
    disc = B*B - 4*A*C
    if disc < 0:
        return -1
    sqrt_disc = math.sqrt(disc)
    t_candidates = []
    t1 = (-B + sqrt_disc)/(2*A)
    if t1 >= 0:
        t_candidates.append(t1)
    t2 = (-B - sqrt_disc)/(2*A)
    if t2 >= 0:
        t_candidates.append(t2)
    if len(t_candidates) == 0:
        return -1
    t_impact = min(t_candidates)

    # intersection coordinates
    xi = x0 + dx*t_impact
    yi = y0 + dy*t_impact
    zi = z0 + dz*t_impact

    # check z range
    z_min = -(nr_rings - 1)/2.0 * ring_gap
    z_max =  (nr_rings - 1)/2.0 * ring_gap
    if (zi < z_min) or (zi > z_max):
        return -1

    # ring index
    ring_float = (zi - z_min) / ring_gap
    ring_i = int(round(ring_float))
    if ring_i < 0 or ring_i >= nr_rings:
        return -1

    # angle
    phi = math.atan2(yi, xi)
    if phi < 0:
        phi += 2*math.pi
    crystal_float = nr_crystals_per_ring * (phi / (2*math.pi))
    crystal_i = int(round(crystal_float)) % nr_crystals_per_ring

    crystal_id = ring_i*nr_crystals_per_ring + crystal_i
    return crystal_id

##########################
# 5) Main event simulation
##########################
def simulate_events(
    n_events,
    emission_data,     # 3D array (128,128,128)
    voxel_size,        # (2.78,2.78,2.78)
    info,
    detector_positions,  # shape (N_crystals, 3)
    detector_ids
):
    """
    Return a NumPy array with columns:
      detID1, detID2, det1X, det1Y, det1Z, det2X, det2Y, det2Z, eventX, eventY, eventZ
    """
    nx, ny, nz = emission_data.shape
    pdf = emission_data.ravel().astype(np.float64)
    pdf_sum = pdf.sum()
    if pdf_sum <= 0:
        raise ValueError("Emission data has non-positive sum.")
    pdf /= pdf_sum  # normalized

    half_x = nx/2.0
    half_y = ny/2.0
    half_z = nz/2.0
    sx, sy, sz = voxel_size

    recorded = []

    for _ in tqdm(range(n_events), desc="Simulating PET events"):
        # pick voxel
        idx_1d = np.random.choice(len(pdf), p=pdf)
        iz = idx_1d // (nx*ny)
        rem = idx_1d % (nx*ny)
        iy = rem // nx
        ix = rem % nx

        # voxel center (the annihilation position)
        x0 = (ix - half_x + 0.5)*sx
        y0 = (iy - half_y + 0.5)*sy
        z0 = (iz - half_z + 0.5)*sz

        # random directions
        dirA = random_unit_3d()
        dirB = -dirA

        # find crystals
        cA = find_crystal_id(x0, y0, z0, dirA, info)
        if cA < 0:
            continue
        cB = find_crystal_id(x0, y0, z0, dirB, info)
        if cB < 0:
            continue

        # get the 3D positions of these detector crystals
        # cA, cB are indices in [0..N_crystals-1].
        # Make sure cA,cB are valid indexes in detector_positions
        if cA >= len(detector_positions) or cB >= len(detector_positions):
            continue

        detA_pos = detector_positions[cA]  # shape (3,)
        detB_pos = detector_positions[cB]  # shape (3,)

        # record:
        # [detID1, detID2, det1x, det1y, det1z, det2x, det2y, det2z, eventx, eventy, eventz]
        recorded.append([
            cA, cB,
            detA_pos[0], detA_pos[1], detA_pos[2],
            detB_pos[0], detB_pos[1], detB_pos[2],
            x0, y0, z0
        ])

    dtype_list = [
        ('detID1','i4'), ('detID2','i4'),
        ('det1x','f4'), ('det1y','f4'), ('det1z','f4'),
        ('det2x','f4'), ('det2y','f4'), ('det2z','f4'),
        ('eventx','f4'), ('eventy','f4'), ('eventz','f4')
    ]
    # Convert to structured array or just a float array
    # We'll output as a float array with a few int columns for convenience.
    array_out = np.array(recorded, dtype=np.float32)
    return array_out

##########################
# 6) Main Script
##########################
def main():
    # 1) Load 3D data
    npy_path = "3d_image_2.npy"
    data_3d = np.load(npy_path).astype(np.float32)  # shape (128,128,128)

    # 2) object_meta
    voxel_size = np.array([2.78, 2.78, 2.78], dtype=np.float32)  # from object_meta

    # 3) Build or load detector geometry
    detector_positions, detector_ids = build_detector_positions(info)
    print(f"Built detector geometry: {len(detector_positions)} crystals.")

    # 4) Decide how many events
    n_events = 2000  # for demonstration, can be 2e6

    # 5) Simulate
    all_data = simulate_events(
        n_events,
        data_3d,
        voxel_size,
        info,
        detector_positions,
        detector_ids
    )
    print(f"Simulated {n_events} events, got {len(all_data)} detections.")

    # 6) Save results: 
    # We want columns:
    #  0=detID1(int), 1=detID2(int),
    #  2=det1x(float), 3=det1y(float), 4=det1z(float),
    #  5=det2x(float), 6=det2y(float), 7=det2z(float),
    #  8=eventx(float), 9=eventy(float), 10=eventz(float)
    # => 11 columns total.
    # We'll write as text:
    #   cA cB cAx cAy cAz cBx cBy cBz x0 y0 z0
    # We'll do 2 integer columns, then 9 float columns.
    # So we can define a format array like:
    fmt_list = ["%d","%d"] + ["%.4f"]*9  # total 11 columns

    # Save to text file
    np.savetxt("listmode_data_extended_chatgpt.txt", all_data, fmt=fmt_list)
    print("Saved extended listmode data to 'listmode_data_extended.txt' with 11 columns.")

if __name__=="__main__":
    main()
