import numpy as np
import gemmi

from ..interfaces import *
from ..dataset import Structure

class AutobuildPreprocessStructure:

    def __init__(self, radius=6.0):
        self.radius=radius
    def __call__(self, structure: StructureInterface, event: EventInterface):
        centroid = np.mean(event.pos_array, axis=0)

        event_centoid = gemmi.Position(
            *centroid
        )

        pdb = structure.structure

        ns = gemmi.NeighborSearch(pdb[0], pdb.cell, self.radius).populate()

        marks = ns.find_atoms(event_centoid, radius=self.radius)
        for mark in marks:
            mark.to_cra(pdb[0]).atom.flag = 's'

        new_structure = gemmi.Structure()

        for model_i, model in enumerate(pdb):
            new_model = gemmi.Model(model.name)
            new_structure.add_model(new_model, pos=-1)

            for chain_i, chain in enumerate(model):
                new_chain = gemmi.Chain(chain.name)
                new_structure[model_i].add_chain(new_chain, pos=-1)

                for residue_i, residue in enumerate(chain):
                    new_residue = gemmi.Residue()
                    new_residue.name = residue.name
                    new_residue.seqid = residue.seqid
                    new_residue.subchain = residue.subchain
                    new_residue.label_seq = residue.label_seq
                    new_residue.het_flag = residue.het_flag
                    new_structure[model_i][chain_i].add_residue(new_residue, pos=-1)

                    for atom_i, atom in enumerate(residue):
                        pos = atom.pos
                        # if pos.dist(event_centoid) > radius:
                        if atom.flag == "s":
                            continue
                        new_structure[model_i][chain_i][residue_i].add_atom(atom, pos=-1)

        for model_i, model in enumerate(pdb):
            pdb.add_model(new_structure[model_i], pos=-1)
            del pdb[0]

        return Structure(None, pdb)