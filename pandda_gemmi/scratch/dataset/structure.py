import dataclasses

import gemmi

from ..interfaces import *
from .. import constants


@dataclasses.dataclass()
class StructurePython:
    string: str

    @staticmethod
    def from_gemmi(structure: gemmi.Structure):
        # json_str = structure.make_mmcif_document().as_json(mmjson=True)
        string = structure.make_minimal_pdb()
        return StructurePython(string)

    def to_gemmi(self):
        structure = gemmi.read_pdb_string(self.string)

        return structure


@dataclasses.dataclass()
class ResidueID:
    model: str
    chain: str
    insertion: str

    @staticmethod
    def from_residue_chain(model: gemmi.Model, chain: gemmi.Chain, res: gemmi.Residue):
        return ResidueID(model.name,
                         chain.name,
                         str(res.seqid.num),
                         )

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return ((self.model, self.chain, self.insertion) ==
                    (other.model, other.chain, other.insertion))
        return NotImplemented

    def __hash__(self):
        return hash((self.model, self.chain, self.insertion))


class Structure:
    def __init__(self, path, structure):
        self.path = path
        self.structure = structure

    @classmethod
    def from_path(cls, path):
        return cls(path, gemmi.read_structure(str(path)))

    def __getitem__(self, item: ResidueID):
        return self.structure[item.model][item.chain][item.insertion]

    def protein_residue_ids(self):
        for model in self.structure:
            for chain in model:
                for residue in chain.get_polymer():
                    if residue.name.upper() not in constants.RESIDUE_NAMES:
                        continue

                    try:
                        has_ca = residue["CA"][0]
                    except Exception as e:
                        continue

                    resid = ResidueID.from_residue_chain(model, chain, residue)
                    yield resid

    def protein_atoms(self):
        for model in self.structure:
            for chain in model:
                for residue in chain.get_polymer():

                    if residue.name.upper() not in constants.RESIDUE_NAMES:
                        continue

                    for atom in residue:
                        yield atom

    def all_atoms(self, exclude_waters=False):
        if exclude_waters:

            for model in self.structure:
                for chain in model:
                    for residue in chain:
                        if residue.is_water():
                            continue

                        for atom in residue:
                            yield atom

        else:
            for model in self.structure:
                for chain in model:
                    for residue in chain:

                        for atom in residue:
                            yield atom

    def __getstate__(self):
        structure_python = StructurePython.from_gemmi(self.structure)
        return (structure_python, self.path)

    def __setstate__(self, data: Tuple[StructurePython, Path]):
        structure_python = data[0]
        path = data[1]
        self.structure = structure_python.to_gemmi()
        self.structure.setup_entities()
        self.path = path
