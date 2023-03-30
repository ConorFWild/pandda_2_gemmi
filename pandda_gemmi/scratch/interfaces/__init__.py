from typing import *
from typing_extensions import ParamSpec, Concatenate, Self
from pathlib import Path

import numpy as np


class ResidueIDInterface(Protocol):
    chain: str
    residue: int
    insertion: str


class LigandFilesInterface(Protocol):
    ligand_cif: Path
    ligand_smiles: Path
    ligand_pdb: Path


class DatasetDirInterface(Protocol):
    dtag: str
    input_pdb_file: Path
    input_mtz_file: Path
    input_ligands: Dict[str, LigandFilesInterface]


class PanDDAInputInterface(Protocol):
    dataset_dirs: Dict[str, DatasetDirInterface]


class PanDDAFSInterface(Protocol):
    input: PanDDAInputInterface


class StructureFactorsInterface(Protocol):
    f: str
    phi: str


class StructureInterface(Protocol):
    def protein_atoms(self):
        ...

    def protein_residue_ids(self):
        ...

    def __getitem__(self, item):
        ...


class ResolutionInterface(Protocol):
    ...


class PositionInterface(Protocol):
    ...


class FractionalInterface(Protocol):
    ...


class SpacegroupInterface(Protocol):
    def xhm(self) -> str:
        ...


GridCoordInterface = Tuple[int, int, int]


class UnitCellInterface(Protocol):
    def fractionalize(self, pos: PositionInterface) -> FractionalInterface:
        ...


class CrystallographicGridInterface(Protocol):
    nu: int
    nv: int
    nw: int
    unit_cell: UnitCellInterface

    def interpolate_value(self, grid_coord: FractionalInterface) -> float:
        ...


class ReflectionsInterface(Protocol):
    path: Path
    reflections: Any
    f: str
    phi: str

    def get_resolution(self) -> float:
        ...

    def transform_f_phi_to_map(self, sample_rate: float = 3.0) -> CrystallographicGridInterface:
        ...


class DatasetInterface(Protocol):
    structure: StructureInterface
    reflections: ReflectionsInterface
    ligand_files: Dict[str, LigandFilesInterface]
    # smoothing_factor: float


class PointPositionArrayInterface(Protocol):
    points: np.ndarray
    positions: np.ndarray


class GridPartitioningInterface(Protocol):
    partitions: Dict[ResidueIDInterface, PointPositionArrayInterface]

class GridMaskInterface(Protocol):
    indicies: np.array

class DFrameInterface(Protocol):
    partitioning: GridPartitioningInterface
    mask: GridMaskInterface

    def get_grid(self) -> CrystallographicGridInterface:
        ...


class TransformInterface(Protocol):
    vec: np.array
    mat: np.array
    # transform: gemmi.Transform
    com_moving: np.array
    com_reference: np.array

    def get_transform(self):
        ...

    def apply_reference_to_moving(self, alignment_positions: Dict[GridCoordInterface, PositionInterface]) -> \
            Dict[GridCoordInterface, PositionInterface]:
        ...

    def apply_moving_to_reference(
            self,
            alignment_positions: Dict[GridCoordInterface, PositionInterface],
    ) -> Dict[GridCoordInterface, PositionInterface]:
        ...

class AlignmentInterface(Protocol):
    transforms: Dict[ResidueIDInterface, TransformInterface]
