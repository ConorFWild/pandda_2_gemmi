from .structure import Structure
from .reflections import Reflections


class XRayDataset:
    def __init__(self, structure, reflections, ligand_files):
        self.structure = structure
        self.reflections = reflections
        self.ligand_files = ligand_files

    @classmethod
    def from_paths(cls, structure_path, reflections_path, ligand_files):
        return XRayDataset(
            Structure.from_path(structure_path),
            Reflections.from_path(reflections_path),
            ligand_files
        )