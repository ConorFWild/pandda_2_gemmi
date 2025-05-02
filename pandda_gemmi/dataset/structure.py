import dataclasses

import gemmi
import numpy as np

from ..interfaces import *


def contains(string, pattern):
    if pattern in string:
        return True
    else:
        return False


@dataclasses.dataclass()
class StructurePython:
    string: str

    @staticmethod
    def from_gemmi(structure: gemmi.Structure):
        string = structure.make_minimal_pdb()
        return StructurePython(string)

    def to_gemmi(self):
        structure = gemmi.read_pdb_string(self.string)

        return structure


@dataclasses.dataclass()
class ResidueID:
    model: int
    chain: str
    number: str

    @staticmethod
    def from_residue_chain(model: gemmi.Model, chain: gemmi.Chain, res: gemmi.Residue):
        return ResidueID(int(model.num),
                         chain.name,
                         str(res.seqid.num),
                         )

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return ((self.model, self.chain, self.number) ==
                    (other.model, other.chain, other.number))
        return NotImplemented

    def __hash__(self):
        return hash((self.model, self.chain, self.number))


def is_protein_residue(residue):
    for atom in residue:
        if "CA" in atom.name.upper():
            return True

    return False


class Structure(StructureInterface):
    def __init__(self, path, structure):
        self.path = path
        self.structure = structure

    @classmethod
    def from_path(cls, path):
        st =gemmi.read_structure(str(path))
        st.setup_entities()
        st.assign_label_seq_id()
        return cls(path, st)

    def __getitem__(self, item: ResidueID):
        return self.structure[item.model][item.chain][item.number]

    def protein_residue_ids(self):
        for model in self.structure:
            for chain in model:
                for residue in chain.get_polymer().first_conformer():
                    if not is_protein_residue(residue):
                        continue

                    resid = ResidueID.from_residue_chain(model, chain, residue)
                    yield resid

    def protein_atoms(self):
        for model in self.structure:
            for chain in model:
                for residue in chain:#"#.get_polymer().first_conformer():

                    if not is_protein_residue(residue):
                        continue

                    for atom in residue:
                        yield atom

    def all_atoms(self, exclude_waters=False):
        if exclude_waters:

            for model in self.structure:
                for chain in model:
                    for residue in chain.first_conformer():
                        if residue.is_water():
                            continue

                        for atom in residue:
                            yield atom

        else:
            for model in self.structure:
                for chain in model:
                    for residue in chain.first_conformer():

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
        self.structure.assign_label_seq_id()

        self.path = path

    def rfree(self):
        return float(self.structure.make_mmcif_document()[0].find_loop("_refine.ls_R_factor_R_free")[0])

    def rwork(self):
        return float(self.structure.make_mmcif_document()[0].find_loop("_refine.ls_R_factor_R_work")[0])


class StructureArray(StructureArrayInterface):
    def __init__(self, models, chains, seq_ids, insertions, atom_ids, positions):

        self.models = np.array(models)
        self.chains = np.array(chains)
        self.seq_ids = np.array(seq_ids)
        self.insertions = np.array(insertions)
        self.atom_ids = np.array(atom_ids)
        self.positions = np.array(positions)

    @classmethod
    def from_structure(cls, structure):
        models = []
        chains = []
        seq_ids = []
        insertions = []
        atom_ids = []
        positions = []
        for model in structure.structure:
            for chain in model:
                for residue in chain.first_conformer():
                    for atom in residue:
                        models.append(model.num)
                        chains.append(chain.name)
                        seq_ids.append(str(residue.seqid.num))
                        insertions.append(residue.seqid.icode)
                        atom_ids.append(atom.name)
                        pos = atom.pos
                        positions.append([pos.x, pos.y, pos.z])

        return cls(models, chains, seq_ids, insertions, atom_ids, positions)

    def mask(self, mask):
        return StructureArray(
            self.models[mask],
            self.chains[mask],
            self.seq_ids[mask],
            self.insertions[mask],
            self.atom_ids[mask],
            self.positions[mask, :]
        )


def save_structure(structure, path):
    structure.structure.write_minimal_pdb(str(path))

def load_structure(path):
    return gemmi.read_structure(str(path))