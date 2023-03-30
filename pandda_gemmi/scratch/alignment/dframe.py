import gemmi
import scipy
import numpy as np

from ..interfaces import *

class PointPositionArray:
    def __init__(self, structure, grid, mask):
        self.points
        self.positions


class StructureArray:
    def __init__(self, structure):
        chains = []
        seq_ids = []
        insertions = []
        atom_ids = []
        positions = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        chains.append(chain.name)
                        seq_ids.append(residue.seqid.num)
                        insertions.append(residue.seqid.icode)
                        atom_ids.append(atom.name)
                        pos = atom.pos
                        positions.append([pos.x, pos.y, pos.z])

        self.chains = np.array(chains)
        self.seq_ids = np.array(seq_ids)
        self.insertions = np.array(insertions)
        self.atom_ids = np.array(atom_ids)
        self.positions = np.array(positions)

        ...


class GridPartitioning:
    def __init__(self, dataset, grid, mask):
        # Get the structure array
        st_array = StructureArray(dataset.structure)

        # Get the tree
        kdtree = scipy.spatial.KDTree(st_array.positions)

        # Get the point array
        point_position_array = PointPositionArray(dataset.structure, grid, mask)

        # Get the NN indexes
        distances, indexes = kdtree.query(point_position_array)

        # Get partions
        self.partitions = {
            (
                st_array.chains[index],
                st_array.residues[index],
                st_array.insertions[index],
            ): point_position_array.points[indexes == index]
            for index
            in indexes
        }


class GridMask:
    def __init__(self, dataset: DatasetInterface, grid, mask_radius=6.0):
        mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        mask.set_unit_cell(grid.unit_cell)
        for atom in dataset.structure.protein_atoms():
            pos = atom.pos
            mask.set_points_around(
                pos,
                radius=mask_radius,
                value=1,
            )
        mask_array = np.array(mask, copy=False, dtype=np.int8)
        self.indicies = np.nonzero(mask_array)


def get_grid_from_dataset(dataset: DatasetInterface):
    return dataset.reflections.transform_f_phi_to_map()


class DFrame:
    def __init__(self, dataset: DatasetInterface, ):
        # Get the grid
        grid = get_grid_from_dataset(dataset)

        # Get the grid parameters
        uc = grid.unit_cell
        self.unit_cell = (uc.a, uc.b, uc.c, uc.alpha, uc.beta, uc.gamma)
        self.spacegroup = gemmi.find_spacegroup_by_name("P 1").number
        self.spacing = (grid.nu, grid.nv, grid.nw)

        # Get the mask
        self.mask = GridMask(dataset, grid)

        # Get the grid partitioning
        self.partitioning = GridPartitioning(dataset, grid, self.mask)
