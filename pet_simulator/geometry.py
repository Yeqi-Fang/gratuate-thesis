# pet_simulator/geometry.py
from dataclasses import dataclass

@dataclass
class PETGeometry:
    """PET scanner geometry parameters"""
    radius: float
    crystals_per_ring: int
    num_rings: int
    crystal_trans_spacing: float
    crystal_axial_spacing: float
    crystal_trans_nr: int
    crystal_axial_nr: int
    module_axial_nr: int
    module_axial_spacing: float

def create_pet_geometry(info: dict) -> PETGeometry:
    """Create PETGeometry from info dictionary"""
    return PETGeometry(
        radius=float(info['radius']),
        crystals_per_ring=int(info['NrCrystalsPerRing']),
        num_rings=int(info['NrRings']),
        crystal_trans_spacing=float(info['crystalTransSpacing']),
        crystal_axial_spacing=float(info['crystalAxialSpacing']),
        crystal_trans_nr=int(info['crystalTransNr']),
        crystal_axial_nr=int(info['crystalAxialNr']),
        module_axial_nr=int(info['moduleAxialNr']),
        module_axial_spacing=float(info['moduleAxialSpacing'])
    )
