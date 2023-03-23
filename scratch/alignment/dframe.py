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
        self.chain
        self.residue
        self.insertion
        self.atom_id
        self.position

        ...


class GridPartitioning:
    def __init__(self, dataset, grid, mask):
        # Get the structure array
        st_array = StructureArray(dataset.structure)

        # Get the tree
        kdtree = scipy.spatial.KDTree(st_array.position)

        # Get the point array
        point_position_array = PointPositionArray(dataset.structure, grid, mask)

        # Get the NN indexes
        distances, indexes = kdtree.query(point_position_array)

        # Get partions
        self.partitions = {
            (
                st_array.chain[index],
                st_array.residue[index],
                st_array.insertion[index],
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
    ...


class DFrame:
    def __init__(self, dataset: DatasetInterface, ):
        # Get the grid parameters
        grid = get_grid_from_dataset(dataset)
        uc = grid.unit_cell
        self.unit_cell = (uc.a, uc.b, uc.c, uc.alpha, uc.beta, uc.gamma)
        self.spacegroup = gemmi.find_spacegroup_by_name("P 1").number
        self.spacing = (grid.nu, grid.nv, grid.nw)

        # Get the mask
        self.mask = GridMask(dataset, grid)

        # Get the grid partitioning
        self.partitioning = GridPartitioning(dataset, grid, self.mask)
