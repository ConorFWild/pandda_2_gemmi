from .structure import Structure
from .reflections import Reflections


class XRayDataset:
    def __init__(self, structure_path, reflections_path, ligand_files):
        self.structure = Structure(structure_path)
        self.reflections = Reflections(reflections_path)
        self.ligand_files = ligand_files
