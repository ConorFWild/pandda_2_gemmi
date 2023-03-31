import itertools

import gemmi
import scipy
import numpy as np

from ..interfaces import *
from ..dataset import ResidueID


class PointPositionArray:
    def __init__(self, points, positions):
        self.points = points
        self.positions = positions

    @staticmethod
    def fractionalize_grid_point_array(grid_point_array, grid):
        return grid_point_array / np.array([grid.nu, grid.nv, grid.nw])

    @staticmethod
    def orthogonalize_fractional_array(fractional_array, grid):
        orthogonalization_matrix = np.array(grid.unit_cell.orthogonalization_matrix.tolist())
        orthogonal_array = np.matmul(orthogonalization_matrix, fractional_array.T).T

        return orthogonal_array

    @staticmethod
    def get_nearby_grid_points(grid, position, radius):
        # Get the fractional position
        # print(f"##########")

        x, y, z = position.x, position.y, position.z

        corners = []
        for dx, dy, dz in itertools.product([-radius, + radius], [-radius, + radius], [-radius, + radius]):
            corner = gemmi.Position(x + dx, y + dy, z + dz)
            corner_fractional = grid.unit_cell.fractionalize(corner)
            corners.append([corner_fractional.x, corner_fractional.y, corner_fractional.z])

        fractional_corner_array = np.array(corners)
        fractional_min = np.min(fractional_corner_array, axis=0)
        fractional_max = np.max(fractional_corner_array, axis=0)

        # print(f"Fractional min: {fractional_min}")
        # print(f"Fractional max: {fractional_max}")

        # Find the fractional bounding box
        # x, y, z = fractional.x, fractional.y, fractional.z
        # dx = radius / grid.nu
        # dy = radius / grid.nv
        # dz = radius / grid.nw
        #
        # # Find the grid bounding box
        # u0 = np.floor((x - dx) * grid.nu)
        # u1 = np.ceil((x + dx) * grid.nu)
        # v0 = np.floor((y - dy) * grid.nv)
        # v1 = np.ceil((y + dy) * grid.nv)
        # w0 = np.floor((z - dz) * grid.nw)
        # w1 = np.ceil((z + dz) * grid.nw)
        u0 = np.floor(fractional_min[0] * grid.nu)
        u1 = np.ceil(fractional_max[0] * grid.nu)
        v0 = np.floor(fractional_min[1] * grid.nv)
        v1 = np.ceil(fractional_max[1] * grid.nv)
        w0 = np.floor(fractional_min[2] * grid.nw)
        w1 = np.ceil(fractional_max[2] * grid.nw)

        # print(f"Fractional bounds are: u: {u0} {u1} : v: {v0} {v1} : w: {w0} {w1}")

        # Get the grid points
        grid_point_array = np.array(
            [
                xyz_tuple
                for xyz_tuple
                in itertools.product(
                np.arange(u0, u1 + 1),
                np.arange(v0, v1 + 1),
                np.arange(w0, w1 + 1),
            )
            ]
        )
        # print(f"Grid point array shape: {grid_point_array.shape}")
        # print(f"Grid point first element: {grid_point_array[0, :]}")

        # Get the point positions
        position_array = PointPositionArray.orthogonalize_fractional_array(
            PointPositionArray.fractionalize_grid_point_array(
                grid_point_array,
                grid,
            ),
            grid,
        )
        # print(f"Grid position array shape: {position_array.shape}")
        # print(f"Grid position first element: {position_array[0, :]}")

        # Get the distances to the position
        distance_array = np.linalg.norm(
            position_array - np.array([position.x, position.y, position.z]),
            axis=1,
        )
        # print(f"Distance array shape: {distance_array.shape}")
        # print(f"Distance array first element: {distance_array[0]}")

        # Mask the points on distances
        points_within_radius = grid_point_array[distance_array < radius]
        positions_within_radius = position_array[distance_array < radius]
        # print(f"Had {grid_point_array.shape} points, of which {points_within_radius.shape} within radius")

        # Bounding box orth
        # orth_bounds_min = np.min(positions_within_radius, axis=0)
        # orth_bounds_max = np.max(positions_within_radius, axis=0)
        # point_bounds_min = np.min(points_within_radius, axis=0)
        # point_bounds_max = np.max(points_within_radius, axis=0)

        # print(f"Original position was: {position.x} {position.y} {position.z}")
        # print(f"Orth bounding box min: {orth_bounds_min}")
        # print(f"Orth bounding box max: {orth_bounds_max}")
        # print(f"Point bounding box min: {point_bounds_min}")
        # print(f"Point bounding box max: {point_bounds_max}")
        # print(f"First point pos pair: {points_within_radius[0, :]} {positions_within_radius[0, :]}")
        # print(f"Last point pos pair: {points_within_radius[-1, :]} {positions_within_radius[-1, :]}")

        return points_within_radius, positions_within_radius

    @staticmethod
    def get_grid_points_around_protein(st: StructureInterface, grid, radius):
        point_arrays = []
        position_arrays = []
        for atom in st.protein_atoms():
            point_array, position_array = PointPositionArray.get_nearby_grid_points(
                grid,
                atom.pos,
                radius
            )
            point_arrays.append(point_array)
            position_arrays.append(position_array)

        all_points_array = np.concatenate(point_arrays, axis=0)
        all_positions_array = np.concatenate(position_arrays, axis=0)

        # print(f"All points shape: {all_points_array.shape}")
        # print(f"All positions shape: {all_positions_array.shape}")

        unique_points, indexes = np.unique(all_points_array, axis=0, return_index=True)
        unique_positions = all_positions_array[indexes, :]
        # print(f"Unique points shape: {unique_points.shape}")
        # print(f"Unique positions shape: {unique_positions.shape}")

        return unique_points, unique_positions

    @classmethod
    def from_structure(cls, st: StructureInterface, grid, radius: float = 6.0):
        point_array, position_array = PointPositionArray.get_grid_points_around_protein(st, grid, radius)
        return PointPositionArray(point_array, position_array)


class StructureArray:
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
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        models.append(model.name)
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
            self.positions[mask,:]
        )


class GridPartitioning:
    def __init__(self, dataset, grid, ):
        # Get the structure array
        st_array = StructureArray.from_structure(dataset.structure)

        # CA point_position_array
        ca_point_position_array = st_array.mask(st_array.atom_ids == "CA")

        # Get the tree
        kdtree = scipy.spatial.KDTree(ca_point_position_array.positions)

        # Get the point array
        point_position_array = PointPositionArray.from_structure(dataset.structure, grid, )

        # Get the NN indexes
        distances, indexes = kdtree.query(point_position_array.positions)

        # Get partions
        self.partitions = {
            ResidueID(st_array.models[index], st_array.chains[index], st_array.seq_ids[index], ): PointPositionArray(
                point_position_array.points[indexes == index],
                point_position_array.positions[indexes == index]
            )
            for index
            in np.unique(indexes)
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
        self.partitioning = GridPartitioning(dataset, grid, )

    def get_grid(self):
        grid = gemmi.FloatGrid(*self.spacing)
        grid.set_unit_cell(gemmi.UnitCell(*self.unit_cell))
        grid.spacegroup = gemmi.find_spacegroup_by_number(self.spacegroup)
        grid_array = np.array(grid, copy=False)
        grid_array[:, :, :] = 0
        return grid
