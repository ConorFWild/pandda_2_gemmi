import itertools
import time

import gemmi
import scipy
import numpy as np

from ..interfaces import *
from ..dataset import ResidueID, StructureArray, contains
from ..dmaps import SparseDMap
from ..dataset import Structure


def transform_structure_to_unit_cell(
        structure,
        unit_cell,
        offset
):
    st = structure.structure.clone()

    transform = gemmi.Transform()
    transform.vec.fromlist(offset.tolist())

    old_poss = []
    new_poss = []
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos = atom.pos
                    old_poss.append([pos.x, pos.y, pos.z])
                    new_pos_vec = transform.apply(pos)
                    new_pos = gemmi.Position(new_pos_vec.x, new_pos_vec.y, new_pos_vec.z)
                    atom.pos = new_pos
                    new_poss.append([new_pos.x, new_pos.y, new_pos.z])

    st.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    st.cell = unit_cell

    return Structure(structure.path, st)


class PointPositionArray(PointPositionArrayInterface):
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
    def fractionalize_orthogonal_array(fractional_array, grid):
        fractionalization_matrix = np.array(grid.unit_cell.fractionalization_matrix.tolist())
        fractional_array = np.matmul(fractionalization_matrix, fractional_array.T).T

        return fractional_array

    @staticmethod
    def fractionalize_grid_point_array_mat(grid_point_array, spacing):
        return grid_point_array / np.array(spacing)

    @staticmethod
    def orthogonalize_fractional_array_mat(fractional_array, orthogonalization_matrix):
        orthogonal_array = np.matmul(orthogonalization_matrix, fractional_array.T).T

        return orthogonal_array

    @staticmethod
    def fractionalize_orthogonal_array_mat(orthogonal_array, fractionalization_matrix):
        fractional_array = np.matmul(fractionalization_matrix, orthogonal_array.T).T

        return fractional_array

    @staticmethod
    def get_nearby_grid_points(grid, position, radius):
        # Get the fractional position

        x, y, z = position.x, position.y, position.z

        corners = []
        for dx, dy, dz in itertools.product([-radius, + radius], [-radius, + radius], [-radius, + radius]):
            corner = gemmi.Position(x + dx, y + dy, z + dz)
            corner_fractional = grid.unit_cell.fractionalize(corner)

            corners.append([corner_fractional.x, corner_fractional.y, corner_fractional.z])

        fractional_corner_array = np.array(corners)
        fractional_min = np.min(fractional_corner_array, axis=0)
        fractional_max = np.max(fractional_corner_array, axis=0)
        # Find the fractional bounding box

        # # Find the grid bounding box

        u0 = np.floor(fractional_min[0] * grid.nu)
        u1 = np.ceil(fractional_max[0] * grid.nu)
        v0 = np.floor(fractional_min[1] * grid.nv)
        v1 = np.ceil(fractional_max[1] * grid.nv)
        w0 = np.floor(fractional_min[2] * grid.nw)
        w1 = np.ceil(fractional_max[2] * grid.nw)

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

        # Get the point positions
        position_array = PointPositionArray.orthogonalize_fractional_array(
            PointPositionArray.fractionalize_grid_point_array(
                grid_point_array,
                grid
            ),
            grid
        )

        # Get the distances to the position
        distance_array = np.linalg.norm(
            position_array - np.array([position.x, position.y, position.z]),
            axis=1,
        )

        # Mask the points on distances
        points_within_radius = grid_point_array[distance_array < radius]
        positions_within_radius = position_array[distance_array < radius]

        return points_within_radius.astype(int), positions_within_radius

    @staticmethod
    def get_nearby_grid_points_parallel(
            spacing,
            fractionalization_matrix,
            orthogonalization_matrix,
            pos_array,
            position,
            radius,
    ):
        # Get the fractional position

        x, y, z = position[0], position[1], position[2]

        corners = []
        for dx, dy, dz in itertools.product([-radius, + radius], [-radius, + radius], [-radius, + radius]):
            corner_fractional = PointPositionArray.fractionalize_orthogonal_array_mat(
                np.array([x + dx, y + dy, z + dz]).reshape((1, 3)),
                fractionalization_matrix,
            )
            corners.append([corner_fractional[0, 0], corner_fractional[0, 1], corner_fractional[0, 2]])

        fractional_corner_array = np.array(corners)

        fractional_min = np.min(fractional_corner_array, axis=0)
        fractional_max = np.max(fractional_corner_array, axis=0)

        # # Find the grid bounding box

        u0 = np.floor(fractional_min[0] * spacing[0])
        u1 = np.ceil(fractional_max[0] * spacing[0])
        v0 = np.floor(fractional_min[1] * spacing[1])
        v1 = np.ceil(fractional_max[1] * spacing[1])
        w0 = np.floor(fractional_min[2] * spacing[2])
        w1 = np.ceil(fractional_max[2] * spacing[2])

        # Get the grid points

        grid = np.mgrid[u0:u1 + 1, v0: v1 + 1, w0:w1 + 1].astype(int)
        grid_point_array = np.hstack([grid[_j].reshape((-1, 1)) for _j in (0, 1, 2)])

        # Get the point positions
        mod_point_array = np.mod(grid_point_array, spacing)
        mod_point_indexes = (
            mod_point_array[:, 0].flatten(), mod_point_array[:, 1].flatten(), mod_point_array[:, 2].flatten())
        position_array = np.zeros(grid_point_array.shape)

        position_array[:, 0] = pos_array[0][mod_point_indexes]
        position_array[:, 1] = pos_array[1][mod_point_indexes]
        position_array[:, 2] = pos_array[2][mod_point_indexes]

        # Get the distances to the position
        distance_array = np.linalg.norm(
            position_array - np.array(position),
            axis=1,
        )

        # Mask the points on distances
        points_within_radius = grid_point_array[distance_array < radius]
        positions_within_radius = position_array[distance_array < radius]

        return points_within_radius.astype(int), positions_within_radius

    @staticmethod
    def get_nearby_grid_points_vectorized(grid, position_array, radius):
        # Get the fractional position

        corners = []
        for dx, dy, dz in itertools.product([-radius, + radius], [-radius, + radius], [-radius, + radius]):
            corner = position_array + np.array([dx, dy, dz])
            corner_fractional = PointPositionArray.fractionalize_orthogonal_array(corner, grid)
            corners.append(corner_fractional)

        # Axis: Atom, coord, corner
        fractional_corner_array = np.stack([corner for corner in corners], axis=-1)

        fractional_min = np.min(fractional_corner_array, axis=-1)
        fractional_max = np.max(fractional_corner_array, axis=-1)

        # # Find the grid bounding box

        u0 = np.floor(fractional_min[:, 0] * grid.nu)
        u1 = np.ceil(fractional_max[:, 0] * grid.nu)
        v0 = np.floor(fractional_min[:, 1] * grid.nv)
        v1 = np.ceil(fractional_max[:, 1] * grid.nv)
        w0 = np.floor(fractional_min[:, 2] * grid.nw)
        w1 = np.ceil(fractional_max[:, 2] * grid.nw)

        # Get the grid points
        point_arrays = []
        position_arrays = []
        for j in range(position_array.shape[0]):
            mesh_grid = np.mgrid[u0[j]: u1[j] + 1, v0[j]: v1[j] + 1, w0[j]: w1[j] + 1]
            grid_point_array = np.hstack([
                mesh_grid[0, :, :].reshape((-1, 1)),
                mesh_grid[1, :, :].reshape((-1, 1)),
                mesh_grid[2, :, :].reshape((-1, 1)),
            ]
            )

            # Get the point positions
            points_position_array = PointPositionArray.orthogonalize_fractional_array(
                PointPositionArray.fractionalize_grid_point_array(
                    grid_point_array,
                    grid,
                ),
                grid,
            )

            # Get the distances to the position
            distance_array = np.linalg.norm(
                points_position_array - position_array[j, :],
                axis=1,
            )

            # Mask the points on distances
            points_within_radius = grid_point_array[distance_array < radius]
            positions_within_radius = points_position_array[distance_array < radius]
            point_arrays.append(points_within_radius.astype(int))
            position_arrays.append(positions_within_radius)

        return point_arrays, position_arrays

    @staticmethod
    def get_grid_points_around_protein(st: StructureInterface, grid, radius, processor: ProcessorInterface):

        positions = []

        point_orthogonalization_matrix = np.matmul(
            np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
            np.diag((1 / grid.nu, 1 / grid.nv, 1 / grid.nw,))
        )

        for atom in st.protein_atoms():
            pos = atom.pos
            positions.append([pos.x, pos.y, pos.z])

        pos_array = np.array(positions)

        spacing = np.array([grid.nu, grid.nv, grid.nw])
        fractionalization_matrix = np.array(grid.unit_cell.fractionalization_matrix.tolist())

        pos_max = np.max(pos_array, axis=0) + radius
        pos_min = np.min(pos_array, axis=0) - radius

        corners = []
        for x, y, z in itertools.product(
                [pos_min[0], pos_max[0]],
                [pos_min[1], pos_max[1]],
                [pos_min[2], pos_max[2]],

        ):
            corner = PointPositionArray.fractionalize_orthogonal_array_mat(
                np.array([x, y, z]).reshape((1, 3)),
                fractionalization_matrix,
            )
            corners.append(corner)

        corner_array = np.vstack(corners)

        fractional_min = np.min(corner_array, axis=0)
        fractional_max = np.max(corner_array, axis=0)

        u0 = int(np.floor(fractional_min[0] * spacing[0]))
        u1 = int(np.ceil(fractional_max[0] * spacing[0]))
        v0 = int(np.floor(fractional_min[1] * spacing[1]))
        v1 = int(np.ceil(fractional_max[1] * spacing[1]))
        w0 = int(np.floor(fractional_min[2] * spacing[2]))
        w1 = int(np.ceil(fractional_max[2] * spacing[2]))

        # Get the grid points
        mgrid = np.mgrid[u0:u1 + 1, v0: v1 + 1, w0:w1 + 1].astype(int)

        grid_point_indicies = [mgrid[_j].reshape((-1, 1)) for _j in (0, 1, 2)]

        # Get and mask a transformed structure
        offset_cart = -np.matmul(point_orthogonalization_matrix, np.array([u0, v0, w0]).reshape((3, 1))).flatten()
        shape = mgrid[0].shape

        new_unit_cell = gemmi.UnitCell(
            shape[0] * (grid.unit_cell.a / grid.nu),
            shape[1] * (grid.unit_cell.b / grid.nv),
            shape[2] * (grid.unit_cell.c / grid.nw),
            grid.unit_cell.alpha,
            grid.unit_cell.beta,
            grid.unit_cell.gamma,
        )

        new_structure = transform_structure_to_unit_cell(
            st,
            new_unit_cell,
            offset_cart
        )

        # Get the outer, inner and inner atomic masks

        # TODO: Get the mask of non-unit cell translation symmetries (handled elsewhere) and subtract from all masks
        # Get mask of symmetry points in native unit cell
        sym_mask_native = gemmi.Int8Grid(*shape)
        sym_mask_native.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        sym_mask_native.set_unit_cell(new_unit_cell)

        ops = [op for op in st.structure.find_spacegroup().operations() if op.triplet() != "x,y,z"]

        unit_cell = st.structure.cell
        for atom in new_structure.protein_atoms():
            for op in ops:
                pos = atom.pos
                pos_frac = unit_cell.fractionalize(pos)
                pos_vec = op.apply_to_xyz([pos_frac.x, pos_frac.y, pos_frac.z])
                sympos = gemmi.Position(unit_cell.orthogonalize(gemmi.Fractional(*pos_vec)))
                sym_mask_native.set_points_around(
                    sympos,
                    radius=2.0,
                    value=1,
                )
        sym_mask_native_array = np.array(sym_mask_native, copy=False, dtype=np.int8)
        sym_mask_native_indicies = np.nonzero(sym_mask_native_array)
        print(f"Number of masked unit cell symmetry positions: {sym_mask_native_indicies[0].size}")
        sym_mask_shifted_indicies = (
            sym_mask_native_indicies[0] - u0,
            sym_mask_native_indicies[1] - v0,
            sym_mask_native_indicies[2] - w0,
        )
        sym_mask_shifted_indicies_masks = (
            (sym_mask_shifted_indicies[0] > -1) & (sym_mask_shifted_indicies[0] < shape[0]),
            (sym_mask_shifted_indicies[1] > -1) & (sym_mask_shifted_indicies[1] < shape[1]),
            (sym_mask_shifted_indicies[2] > -1) & (sym_mask_shifted_indicies[2] < shape[2]),
        )
        sym_mask_shifted_indicies_mask = sym_mask_shifted_indicies_masks[0] & sym_mask_shifted_indicies_masks[1] & sym_mask_shifted_indicies_masks[2]
        sym_mask_shifted_indicies_masked = (
            sym_mask_shifted_indicies[0][sym_mask_shifted_indicies_mask],
            sym_mask_shifted_indicies[1][sym_mask_shifted_indicies_mask],
            sym_mask_shifted_indicies[2][sym_mask_shifted_indicies_mask]
        )

        print(f"Symmetry mask bounded to protein cell min/max: ")
        print(f"\t{np.min(sym_mask_shifted_indicies_masked[0])} {np.max(sym_mask_shifted_indicies_masked[0])}")
        print(f"\t{np.min(sym_mask_shifted_indicies_masked[1])} {np.max(sym_mask_shifted_indicies_masked[1])}")
        print(f"\t{np.min(sym_mask_shifted_indicies_masked[2])} {np.max(sym_mask_shifted_indicies_masked[2])}")

        # Shift to new unit cell


        # Outer mask
        outer_mask = gemmi.Int8Grid(*shape)
        outer_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        outer_mask.set_unit_cell(new_unit_cell)

        for atom in new_structure.protein_atoms():
            pos = atom.pos
            outer_mask.set_points_around(
                pos,
                radius=6.0,
                value=1,
            )
        outer_mask_array = np.array(outer_mask, copy=False, dtype=np.int8)
        # TODO: mask out non-translation symmetry points
        print(f"Outer mask size before masking: {np.sum(outer_mask_array)}")
        outer_mask_array[sym_mask_shifted_indicies_masked] = 0
        print(f"Outer mask size after masking: {np.sum(outer_mask_array)}")


        outer_indicies = np.nonzero(outer_mask_array)
        outer_indicies_native = (
            np.mod(outer_indicies[0] + u0, grid.nu),
            np.mod(outer_indicies[1] + v0, grid.nv),
            np.mod(outer_indicies[2] + w0, grid.nw),
        )
        indicies_min = [
            np.min(outer_indicies_native[0]),
            np.min(outer_indicies_native[1]),
            np.min(outer_indicies_native[2]),
        ]
        indicies_max = [
            np.max(outer_indicies_native[0]),
            np.max(outer_indicies_native[1]),
            np.max(outer_indicies_native[2]),
        ]

        inner_mask = gemmi.Int8Grid(*shape)
        inner_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        inner_mask.set_unit_cell(new_unit_cell)
        for atom in new_structure.protein_atoms():
            pos = atom.pos
            inner_mask.set_points_around(
                pos,
                radius=2.0,
                value=1,
            )
        inner_mask_array = np.array(inner_mask, copy=False, dtype=np.int8)
        # TODO: mask out non-translation symmetry points
        inner_mask_array[sym_mask_shifted_indicies_masked] = 0

        inner_indicies = np.nonzero(inner_mask_array)
        inner_indicies_native = (
            np.mod(inner_indicies[0] + u0, grid.nu),
            np.mod(inner_indicies[1] + v0, grid.nv),
            np.mod(inner_indicies[2] + w0, grid.nw),
        )
        indicies_min = [
            np.min(inner_indicies_native[0]),
            np.min(inner_indicies_native[1]),
            np.min(inner_indicies_native[2]),
        ]
        indicies_max = [
            np.max(inner_indicies_native[0]),
            np.max(inner_indicies_native[1]),
            np.max(inner_indicies_native[2]),
        ]

        sparse_inner_indicies = inner_mask_array[outer_indicies] == 1

        inner_atomic_mask = gemmi.Int8Grid(*shape)
        inner_atomic_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        inner_atomic_mask.set_unit_cell(new_unit_cell)
        for atom in new_structure.protein_atoms():
            pos = atom.pos
            inner_atomic_mask.set_points_around(
                pos,
                radius=0.5,
                value=1,
            )
        inner_atomic_mask_array = np.array(inner_atomic_mask, copy=False, dtype=np.int8)
        # TODO: mask out non-translation symmetry points
        inner_atomic_mask_array[sym_mask_shifted_indicies_masked] = 0

        inner_atomic_indicies = np.nonzero(inner_atomic_mask_array)
        inner_atomic_indicies_native = (
            np.mod(inner_atomic_indicies[0] + u0, grid.nu),
            np.mod(inner_atomic_indicies[1] + v0, grid.nv),
            np.mod(inner_atomic_indicies[2] + w0, grid.nw),
        )
        indicies_min = [
            np.min(inner_atomic_indicies_native[0]),
            np.min(inner_atomic_indicies_native[1]),
            np.min(inner_atomic_indicies_native[2]),
        ]
        indicies_max = [
            np.max(inner_atomic_indicies_native[0]),
            np.max(inner_atomic_indicies_native[1]),
            np.max(inner_atomic_indicies_native[2]),
        ]

        sparse_inner_atomic_indicies = inner_atomic_mask_array[outer_indicies] == 1

        all_indicies = {
            "outer": outer_indicies_native,
            "inner": inner_indicies_native,
            "inner_sparse": sparse_inner_indicies,
            "atomic": inner_atomic_indicies_native,
            "atomic_sparse": sparse_inner_atomic_indicies
        }

        # Get the grid points (real space, not modulus!) in the mask
        shifted_grid_point_indicies = tuple(
            (grid_point_indicies[_j] - np.array([u0, v0, w0]).astype(np.int)[_j]).flatten()
            for _j
            in (0, 1, 2)
        )
        grid_point_indicies_mask = outer_mask_array[shifted_grid_point_indicies] == 1

        grid_point_array = np.vstack(
            [
                grid_point_indicies[_j][grid_point_indicies_mask].flatten()
                for _j
                in (0, 1, 2)
            ]
        ).astype(np.int)

        unique_points = grid_point_array.T

        unique_positions = np.matmul(point_orthogonalization_matrix, grid_point_array).T

        return unique_points, unique_positions, all_indicies

    @classmethod
    def from_structure(cls, st: StructureInterface, grid, processor, radius: float = 6.0):
        point_array, position_array, all_indicies = PointPositionArray.get_grid_points_around_protein(st, grid, radius,
                                                                                                      processor)
        return PointPositionArray(point_array, position_array), all_indicies

def get_nearby_symmetry_atoms_pos_array(structure, structure_array):

    # Get the unit cell
    cell = structure.structure.cell

    # Get the array of symmetry positions and transform to homogenous coordinates
    st_array = structure_array.positions.T
    st_array_homogeous = np.concatenate(
        [
            st_array,
            np.zeros((1, st_array.shape[1])) + 1
        ],
        axis=0
    )
    print(f"Structure array homogeous shape: {st_array_homogeous.shape}")

    # Get a list of transformation matricies for symmetry ops
    ops = [op for op in structure.structure.find_spacegroup().operations() ]
    symops = []
    for dx, dy, dz in itertools.product([-1,0,1], [-1,0,1], [-1,0,1], ):

        for op in ops:
            if (dy == 0) & (dy == 0) & (dz == 0):
                if op.triplet() == "x,y,z":
                    continue

            fractional_seitz = np.array(op.float_seitz())
            fractional_seitz[0, -1] = (fractional_seitz[0,-1] + dx) * cell.a
            fractional_seitz[1, -1] = (fractional_seitz[1, -1] + dy) * cell.b
            fractional_seitz[2, -1] = (fractional_seitz[2, -1] + dz) * cell.c
            symops.append(fractional_seitz)
            # rotmat = np.array(op.rot, dtype=np.float32) / 24.0
            # transmat =
            # ...

    print(f"Symmatrix shape: {symops[0].shape}")

    # Generate each symmetry image of the structure array
    symatoms_list = []
    for symop in symops:
        symatoms_list.append(
            np.matmul(symop, st_array_homogeous).T
        )

    # Concatenate the symmetry images
    symatoms_homogeous = np.concatenate(symatoms_list, axis=0)
    print(f"Symatoms shape before dropping homogenising factor: {symatoms_homogeous.shape}")


    # Go back to cartesian coordinates
    symatoms = symatoms_homogeous[:, :-1]
    print(f"Symatoms shape after dropping homogenising factor: {symatoms.shape}")

    # Get those in a box bounding the structure + mask radius
    pos_min = np.min(st_array, axis=1) - np.array([6.0,6.0,6.0])
    pos_max = np.max(st_array, axis=1) + np.array([6.0,6.0,6.0])
    print(f"Min and max of structure array: {pos_min} {pos_max}")

    mask = (symatoms[:,0] > pos_min[0]) & (symatoms[:,0] < pos_max[0]) & (symatoms[:,1] > pos_min[1]) & (symatoms[:,1] < pos_max[1]) & (symatoms[:,2] > pos_min[2]) & (symatoms[:,2] < pos_max[2])
    print(f"Mask shape: {mask.shape}")

    nearby_symatoms = symatoms[mask]
    print(f"Nearby symatoms shape: {nearby_symatoms.shape}")


    return nearby_symatoms
    #


class GridPartitioning(GridPartitioningInterface):
    def __init__(self, partitions):
        self.partitions = partitions

    @classmethod
    def from_dataset(cls, dataset, grid, processor):
        # Get the structure array
        st_array = StructureArray.from_structure(dataset.structure)

        # CA point_position_array
        used_insertions = []
        ca_mask = []
        for j, atom_id in enumerate(st_array.atom_ids):
            key = (st_array.chains[j], st_array.seq_ids[j])
            if (key not in used_insertions) and contains(str(atom_id).upper(), "CA"):
                ca_mask.append(True)
                used_insertions.append(key)
            else:
                ca_mask.append(False)

        ca_point_position_array = st_array.mask(np.array(ca_mask))

        # Get the nearby symmetry atoms
        nearby_symmetry_atom_pos_array = get_nearby_symmetry_atoms_pos_array(dataset.structure, st_array)
        print(f"Got nearby symmetry poss: {nearby_symmetry_atom_pos_array.shape}")

        # Get the search atoms
        search_atom_poss = np.concatenate(
            [
                ca_point_position_array.positions,
                nearby_symmetry_atom_pos_array
            ],
            axis=0
        )

        # Get the tree
        kdtree = scipy.spatial.KDTree(
            # ca_point_position_array.positions
            search_atom_poss
        )

        # Get the pointposition array
        point_position_array, all_indicies = PointPositionArray.from_structure(
            dataset.structure,
            grid,
            processor,
        )

        # Get the NN indexes
        distances, indexes = kdtree.query(point_position_array.positions, workers=12)

        print(f"Got {indexes[indexes >= ca_point_position_array.positions.shape[0]]} points associated with symmetry atoms")

        # Deal with unit cell translation symmetry duplicated indicies
        # TODO: Get grid space duplicate indicies i.e. ones for which the unit cell modulus is the same
        # TODO: and mask the one that is further from its respective CA

        ##


        # Construct the partition
        partitions = {
            ResidueID(
                ca_point_position_array.models[index],
                ca_point_position_array.chains[index],
                ca_point_position_array.seq_ids[index],
            ): PointPositionArray(
                point_position_array.points[indexes == index],
                point_position_array.positions[indexes == index]
            )
            for index
            in np.unique(indexes)
            if index < ca_point_position_array.positions.shape[0]
        }

        return cls(partitions, ), all_indicies


class GridMask(GridMaskInterface):
    def __init__(self, indicies, indicies_inner, indicies_sparse_inner, indicies_inner_atomic,
                 indicies_sparse_inner_atomic):
        self.indicies = indicies
        self.indicies_inner = indicies_inner
        self.indicies_sparse_inner = indicies_sparse_inner
        self.indicies_inner_atomic = indicies_inner_atomic
        self.indicies_sparse_inner_atomic = indicies_sparse_inner_atomic

    @classmethod
    def from_dataset(cls, dataset: DatasetInterface, grid, mask_radius=6.0, mask_radius_inner=2.0):

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
        indicies = np.nonzero(mask_array)

        mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        mask.set_unit_cell(grid.unit_cell)
        for atom in dataset.structure.protein_atoms():
            pos = atom.pos
            mask.set_points_around(
                pos,
                radius=mask_radius_inner,
                value=1,
            )
        mask_array = np.array(mask, copy=False, dtype=np.int8)
        indicies_inner = np.nonzero(mask_array)
        indicies_sparse_inner = mask_array[indicies] == 1.0

        mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        mask.set_unit_cell(grid.unit_cell)
        for atom in dataset.structure.protein_atoms():
            pos = atom.pos
            mask.set_points_around(
                pos,
                radius=0.5,
                value=1,
            )
        mask_array = np.array(mask, copy=False, dtype=np.int8)
        indicies_inner_atomic = np.nonzero(mask_array)
        indicies_sparse_inner_atomic = mask_array[indicies] == 1.0

        return cls(indicies, indicies_inner, indicies_sparse_inner, indicies_inner_atomic, indicies_sparse_inner_atomic)

    @classmethod
    def from_indicies(cls, all_indicies):
        return cls(
            all_indicies["outer"],
            all_indicies["inner"],
            all_indicies["inner_sparse"],
            all_indicies["atomic"],
            all_indicies["atomic_sparse"]
        )


def get_grid_from_dataset(dataset: DatasetInterface):
    return dataset.reflections.transform_f_phi_to_map()


class DFrame:
    def __init__(self, dataset: DatasetInterface, processor):
        # Get the grid
        grid = get_grid_from_dataset(dataset)

        # Get the grid parameters
        uc = grid.unit_cell
        self.unit_cell = (uc.a, uc.b, uc.c, uc.alpha, uc.beta, uc.gamma)
        self.spacegroup = gemmi.find_spacegroup_by_name("P 1").number
        self.spacing = (grid.nu, grid.nv, grid.nw)

        # Get the grid partitioning
        begin_partition = time.time()
        self.partitioning, all_indicies = GridPartitioning.from_dataset(dataset, grid, processor)
        finish_partition = time.time()
        print(f"\tGot Partitions in {finish_partition - begin_partition}")

        # Get the mask
        begin_mask = time.time()
        self.mask = GridMask.from_indicies(all_indicies)
        finish_mask = time.time()
        print(f"\tGot mask in {finish_mask - begin_mask}")

    def get_grid(self):
        grid = gemmi.FloatGrid(*self.spacing)
        grid.set_unit_cell(gemmi.UnitCell(*self.unit_cell))
        grid.spacegroup = gemmi.find_spacegroup_by_number(self.spacegroup)
        grid_array = np.array(grid, copy=False)
        grid_array[:, :, :] = 0.0
        return grid

    def unmask(self, sparse_dmap, ):
        grid = self.get_grid()
        grid_array = np.array(grid, copy=False)
        grid_array[self.mask.indicies] = sparse_dmap.data
        return grid

    def unmask_inner(self, sparse_dmap, ):
        grid = self.get_grid()
        grid_array = np.array(grid, copy=False)
        grid_array[self.mask.indicies_inner] = sparse_dmap.data
        return grid

    def mask_grid(self, grid):
        grid_array = np.array(grid, copy=False)
        data = grid_array[self.mask.indicies]
        return SparseDMap(data)

    def mask_inner(self, grid):
        grid_array = np.array(grid, copy=False)
        data = grid_array[self.mask.indicies_inner]
        return SparseDMap(data)
