from __future__ import annotations

import typing
import dataclasses

import itertools
from pathlib import Path

import gemmi
import numpy as np
from scipy import spatial
from joblib.externals.loky import set_loky_pickler
from pandda_gemmi.analyse_interface import GetGridInterface

set_loky_pickler('pickle')

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.constants import *
from pandda_gemmi.python_types import *
from pandda_gemmi.dataset import ResidueID, Reference, Structure, Symops


@dataclasses.dataclass()
class Partitioning(PartitioningInterface):
    partitioning: typing.Dict[ResidueIDInterface, typing.Dict[GridCoordInterface, PositionInterface]]
    protein_mask: gemmi.Int8Grid
    inner_mask: gemmi.Int8Grid
    contact_mask: gemmi.Int8Grid
    symmetry_mask: gemmi.Int8Grid
    total_mask: np.ndarray

    def __getitem__(self, item: ResidueIDInterface):
        return self.partitioning[item]

    def __iter__(self) -> Iterator[ResidueIDInterface]:
        for residue_id in self.partitioning:
            yield residue_id

    @staticmethod
    def from_reference(reference: ReferenceInterface,
                       grid: gemmi.FloatGrid,
                       mask_radius: float,
                       mask_radius_symmetry: float,
                       debug: Debug = Debug.DEFAULT
                       ):

        return Partitioning.from_structure(reference.dataset.structure,
                                           grid,
                                           mask_radius,
                                           mask_radius_symmetry,
                                           debug=debug
                                           )

    @staticmethod
    def get_coord_tuple(grid, ca_position_array, structure: Structure, mask_radius: float = 6.0, buffer: float = 3.0,
                        debug: Debug = Debug.DEFAULT):
        # Get the bounds
        min_x = ca_position_array[:, 0].min() - buffer
        max_x = ca_position_array[:, 0].max() + buffer
        min_y = ca_position_array[:, 1].min() - buffer
        max_y = ca_position_array[:, 1].max() + buffer
        min_z = ca_position_array[:, 2].min() - buffer
        max_z = ca_position_array[:, 2].max() + buffer

        # Get the upper and lower bounds of the grid as positions
        grid_min_cart = gemmi.Position(min_x, min_y, min_z)
        grid_max_cart = gemmi.Position(max_x, max_y, max_z)

        # Get them as fractions of the unit cell
        grid_min_frac = grid.unit_cell.fractionalize(grid_min_cart)
        grid_max_frac = grid.unit_cell.fractionalize(grid_max_cart)
        if debug >= Debug.PRINT_NUMERICS:
            print(f"Min grid cart: {grid_min_cart}")
            print(f"Max grid cart: {grid_max_cart}")

        # Get them as coords
        grid_min_coord = [int(grid_min_frac[0] * grid.nu), int(grid_min_frac[1] * grid.nv),
                          int(grid_min_frac[2] * grid.nw), ]
        grid_max_coord = [int(grid_max_frac[0] * grid.nu), int(grid_max_frac[1] * grid.nv),
                          int(grid_max_frac[2] * grid.nw), ]

        if debug >= Debug.PRINT_NUMERICS:
            print(f"Min grid coord: {grid_min_coord}")
            print(f"Max grid coord: {grid_max_coord}")

        # Get these as fractions
        # fractional_grid_min = [
        #     grid.unit_cell.fractionalize(grid_min_coord[0]),
        #     grid.unit_cell.fractionalize(grid_min_coord[1]),
        #     grid.unit_cell.fractionalize(grid_min_coord[2]),
        # ]
        fractional_grid_min = [
            grid_min_coord[0] * (1.0 / grid.nu),
            grid_min_coord[1] * (1.0 / grid.nv),
            grid_min_coord[2] * (1.0 / grid.nw),
        ]
        # fractional_grid_max = [
        #     grid.unit_cell.fractionalize(grid_max_coord[0]),
        #     grid.unit_cell.fractionalize(grid_max_coord[1]),
        #     grid.unit_cell.fractionalize(grid_max_coord[2]),
        #     ]
        fractional_grid_max = [
            grid_max_coord[0] * (1.0 / grid.nu),
            grid_max_coord[1] * (1.0 / grid.nv),
            grid_max_coord[2] * (1.0 / grid.nw),
        ]
        fractional_diff = [
            fractional_grid_max[0] - fractional_grid_min[0],
            fractional_grid_max[1] - fractional_grid_min[1],
            fractional_grid_max[2] - fractional_grid_min[2],
        ]
        if debug >= Debug.PRINT_NUMERICS:
            print(f"Min grid frac: {fractional_grid_min}")
            print(f"Max grid frac: {fractional_grid_max}")

        # Get the grid of points around the protein

        coord_product = itertools.product(
            range(grid_min_coord[0], grid_max_coord[0]),
            range(grid_min_coord[1], grid_max_coord[1]),
            range(grid_min_coord[2], grid_max_coord[2]),
        )

        coord_array = np.array([[x, y, z] for x, y, z in coord_product])

        coord_tuple = (coord_array[:, 0],
                       coord_array[:, 1],
                       coord_array[:, 2],
                       )
        # print((
        #     f"coord_tuple - points in the refence grid\n"
        #     f"min: {np.min(coord_tuple[0])} {np.min(coord_tuple[1])} {np.min(coord_tuple[2])} \n"
        #     f"Max: {np.max(coord_tuple[0])} {np.max(coord_tuple[1])} {np.max(coord_tuple[2])} \n"
        #     f"Length: {coord_tuple[0].shape}"
        # ))

        # Get the corresponding protein grid
        protein_grid = gemmi.Int8Grid(
            grid_max_coord[0] - grid_min_coord[0],
            grid_max_coord[1] - grid_min_coord[1],
            grid_max_coord[2] - grid_min_coord[2],
        )
        protein_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        protein_grid_unit_cell = gemmi.UnitCell(
            grid.unit_cell.a * fractional_diff[0],
            grid.unit_cell.b * fractional_diff[1],
            grid.unit_cell.c * fractional_diff[2],
            grid.unit_cell.alpha,
            grid.unit_cell.beta,
            grid.unit_cell.gamma,
        )
        protein_grid.set_unit_cell(protein_grid_unit_cell)

        # Mask
        for atom in structure.all_atoms():
            pos = atom.pos
            pos_transformed = gemmi.Position(pos.x - grid_min_cart[0],
                                             pos.y - grid_min_cart[1],
                                             pos.z - grid_min_cart[2],
                                             )
            protein_grid.set_points_around(pos_transformed,
                                           radius=mask_radius,
                                           value=1,
                                           )

        # # Get the corresponding unit cell points
        coord_unit_cell_tuple = (np.mod(coord_tuple[0], grid.nu),
                                 np.mod(coord_tuple[1], grid.nv),
                                 np.mod(coord_tuple[2], grid.nw),
                                 )
        # print((
        #     f"coord_unit_cell_tuple - points in the reference grid normalised to the unit cell\n"
        #     f"min: {np.min(coord_unit_cell_tuple[0])} {np.min(coord_unit_cell_tuple[1])} {np.min(coord_unit_cell_tuple[2])} \n"
        #     f"Max: {np.max(coord_unit_cell_tuple[0])} {np.max(coord_unit_cell_tuple[1])} {np.max(coord_unit_cell_tuple[2])} \n"
        #     f"Length: {coord_unit_cell_tuple[0].shape}"
        # ))

        # Get the corresponging protein_grid points
        coord_mask_grid_tuple = (
            coord_tuple[0] - grid_min_coord[0],
            coord_tuple[1] - grid_min_coord[1],
            coord_tuple[2] - grid_min_coord[2],
        )
        # print((
        #     f"coord_mask_grid_tuple - points in the portein frame grid\n"
        #     f"min: {np.min(coord_mask_grid_tuple[0])} {np.min(coord_mask_grid_tuple[1])} {np.min(coord_mask_grid_tuple[2])} \n"
        #     f"Max: {np.max(coord_mask_grid_tuple[0])} {np.max(coord_mask_grid_tuple[1])} {np.max(coord_mask_grid_tuple[2])} \n"
        #     f"Length: {coord_mask_grid_tuple[0].shape}"
        # ))

        # Check which of them are in the mask
        mask_array = np.array(protein_grid, copy=False, dtype=np.int8)
        in_mask_array_int = mask_array[coord_mask_grid_tuple]
        in_mask_array = in_mask_array_int == 1

        # Mask those coords in the tuples
        coord_array_in_mask = (
            coord_tuple[0][in_mask_array],
            coord_tuple[1][in_mask_array],
            coord_tuple[2][in_mask_array],
        )
        # print((
        #     f"coord_array_in_mask - points in the reference grid in the mask\n"
        #     f"min: {np.min(coord_array_in_mask[0])} {np.min(coord_array_in_mask[1])} {np.min(coord_array_in_mask[2])} \n"
        #     f"Max: {np.max(coord_array_in_mask[0])} {np.max(coord_array_in_mask[1])} {np.max(coord_array_in_mask[2])} \n"
        # ))

        coord_array_unit_cell_in_mask = (
            coord_unit_cell_tuple[0][in_mask_array],
            coord_unit_cell_tuple[1][in_mask_array],
            coord_unit_cell_tuple[2][in_mask_array],
        )
        # print((
        #     f"coord_array_unit_cell_in_mask - points in the reference grid nomralise to the unit cell in the mask\n"
        #     f"min: {np.min(coord_array_unit_cell_in_mask[0])} {np.min(coord_array_unit_cell_in_mask[1])} {np.min(coord_array_unit_cell_in_mask[2])} \n"
        #     f"Max: {np.max(coord_array_unit_cell_in_mask[0])} {np.max(coord_array_unit_cell_in_mask[1])} {np.max(coord_array_unit_cell_in_mask[2])} \n"
        # ))

        return coord_array_in_mask, coord_array_unit_cell_in_mask

    @staticmethod
    def get_position_list(mask, coord_array):
        positions = []
        for u, v, w in coord_array:
            point = mask.get_point(u, v, w)
            position = mask.point_to_position(point)
            positions.append((position[0],
                              position[1],
                              position[2],
                              )
                             )
        return positions

    @staticmethod
    def from_structure_multiprocess(structure: StructureInterface,
                                    grid: CrystallographicGridInterface,  #: Grid,
                                    mask_radius: float,
                                    mask_radius_symmetry: float, ):

        return Partitioning.from_structure(structure,
                                           grid,
                                           mask_radius,
                                           mask_radius_symmetry, )

    # @staticmethod
    # def from_structure(structure: StructureInterface,
    #                    grid: CrystallographicGridInterface,
    #                    mask_radius: float,
    #                    mask_radius_symmetry: float,
    #                    debug: Debug = Debug.DEFAULT,
    #                    ):
    #     poss = []
    #     res_indexes = {}
    #
    #     for i, res_id in enumerate(structure.protein_residue_ids()):
    #         res_span = structure[res_id]
    #         res = res_span[0]
    #         ca = res["CA"][0]
    #         orthogonal = ca.pos
    #         poss.append(orthogonal)
    #         res_indexes[i] = res_id
    #
    #     ca_position_array = np.array([[x for x in pos] for pos in poss])
    #
    #     kdtree = spatial.KDTree(ca_position_array)
    #
    #     mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
    #     mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    #     mask.set_unit_cell(grid.unit_cell)
    #     for atom in structure.protein_atoms():
    #         pos = atom.pos
    #         mask.set_points_around(pos,
    #                                radius=mask_radius,
    #                                value=1,
    #                                )
    #     mask_array = np.array(mask, copy=False, dtype=np.int8)
    #
    #     # Get the inner mask
    #     inner_mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
    #     inner_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    #     inner_mask.set_unit_cell(grid.unit_cell)
    #     for atom in structure.protein_atoms():
    #         pos = atom.pos
    #         inner_mask.set_points_around(pos,
    #                                      radius=mask_radius_symmetry,
    #                                      value=1,
    #                                      )
    #     # mask_array = np.array(mask, copy=False, dtype=np.int8)
    #
    #     # Get the contact mask
    #     contact_mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
    #     contact_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    #     contact_mask.set_unit_cell(grid.unit_cell)
    #     for atom in structure.protein_atoms():
    #         pos = atom.pos
    #         contact_mask.set_points_around(pos,
    #                                        radius=4.0,
    #                                        value=1,
    #                                        )
    #     # mask_array = np.array(mask, copy=False, dtype=np.int8)
    #
    #     # Mask the symmetry points
    #     symmetry_mask = Partitioning.get_symmetry_contact_mask(structure, grid, mask, mask_radius_symmetry)
    #     symmetry_mask_array = np.array(symmetry_mask, copy=False, dtype=np.int8)
    #
    #     coord_tuple_source, coord_array_unit_cell_in_mask = Partitioning.get_coord_tuple(
    #         mask,
    #         ca_position_array,
    #         structure,
    #         mask_radius,
    #         debug=debug,
    #     )
    #
    #     # Mask by protein
    #     protein_mask_indicies = mask_array[coord_array_unit_cell_in_mask]
    #
    #     # Mask by symmetry
    #     symmetry_mask_indicies = symmetry_mask_array[coord_array_unit_cell_in_mask]
    #
    #     # Combine masks
    #     combined_indicies = np.zeros(symmetry_mask_indicies.shape)
    #     combined_indicies[protein_mask_indicies == 1] = 1
    #     combined_indicies[symmetry_mask_indicies == 1] = 0
    #
    #     # Resample coords
    #     coord_tuple = (coord_tuple_source[0][combined_indicies == 1],
    #                    coord_tuple_source[1][combined_indicies == 1],
    #                    coord_tuple_source[2][combined_indicies == 1],
    #                    )
    #
    #     #
    #     coord_array = np.concatenate(
    #         [
    #             coord_tuple[0].reshape((-1, 1)),
    #             coord_tuple[1].reshape((-1, 1)),
    #             coord_tuple[2].reshape((-1, 1)),
    #         ],
    #         axis=1,
    #     )
    #
    #     # Get positions
    #     position_list = Partitioning.get_position_list(mask, coord_array)
    #     position_array = np.array(position_list)
    #
    #     distances, indexes = kdtree.query(position_array)
    #
    #     # Get the partitions
    #     partitions = {}
    #     for i, coord_as_array in enumerate(coord_array):
    #         coord = (int(coord_as_array[0]),
    #                  int(coord_as_array[1]),
    #                  int(coord_as_array[2]),
    #                  )
    #         position = position_list[i]
    #
    #         res_num = indexes[i]
    #
    #         res_id = res_indexes[res_num]
    #         if res_id not in partitions:
    #             partitions[res_id] = {}
    #
    #         # partitions[res_id][coord] = gemmi.Position(*position)
    #
    #         coord_unit_cell = (int(coord_array_unit_cell_in_mask[0][i]),
    #                            int(coord_array_unit_cell_in_mask[1][i]),
    #                            int(coord_array_unit_cell_in_mask[2][i]),
    #                            )
    #         partitions[res_id][coord] = position
    #
    #     total_mask = np.zeros(mask_array.shape, dtype=np.int8)
    #     total_mask[
    #         coord_array_unit_cell_in_mask[0][combined_indicies == 1],
    #         coord_array_unit_cell_in_mask[1][combined_indicies == 1],
    #         coord_array_unit_cell_in_mask[2][combined_indicies == 1],
    #     ] = 1
    #
    #     return Partitioning(partitions, mask,
    #                         inner_mask,
    #                         contact_mask,
    #                         symmetry_mask, total_mask)
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
        print(f"##########")

        fractional = grid.unit_cell.fractionalize(position)


        # Find the fractional bounding box
        x, y, z = fractional.x, fractional.y, fractional.z
        dx = radius / grid.nu
        dy = radius / grid.nv
        dz = radius / grid.nw

        # Find the grid bounding box
        u0 = np.floor((x - dx) * grid.nu)
        u1 = np.ceil((x + dx) * grid.nu)
        v0 = np.floor((y - dy) * grid.nv)
        v1 = np.ceil((y + dy) * grid.nv)
        w0 = np.floor((z - dz) * grid.nw)
        w1 = np.ceil((z + dz) * grid.nw)

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
        print(f"Grid point array shape: {grid_point_array.shape}")
        print(f"Grid point first element: {grid_point_array[0, :]}")

        # Get the point positions
        position_array = Partitioning.orthogonalize_fractional_array(
            Partitioning.fractionalize_grid_point_array(
                grid_point_array,
                grid,
            ),
            grid,
        )
        print(f"Grid position array shape: {position_array.shape}")
        print(f"Grid position first element: {position_array[0, :]}")

        # Get the distances to the position
        distance_array = np.linalg.norm(
            position_array - np.array([position.x, position.y, position.z]),
            axis=1,
        )
        print(f"Distance array shape: {distance_array.shape}")
        print(f"Distance array first element: {distance_array[0]}")

        # Mask the points on distances
        points_within_radius = grid_point_array[distance_array < radius]
        positions_within_radius = position_array[distance_array < radius]

        # Bounding box orth
        orth_bounds_min = np.min(positions_within_radius, axis=0)
        orth_bounds_max = np.max(positions_within_radius, axis=0)
        point_bounds_min = np.min(points_within_radius, axis=0)
        point_bounds_max = np.max(points_within_radius, axis=0)

        print(f"Original position was: {position.x} {position.y} {position.z}")
        print(f"Orth bounding box min: {orth_bounds_min}")
        print(f"Orth bounding box max: {orth_bounds_max}")
        print(f"Point bounding box min: {point_bounds_min}")
        print(f"Point bounding box max: {point_bounds_max}")
        print(f"First point pos pair: {positions_within_radius[0,:]} {positions_within_radius[0,:]}")
        print(f"Last point pos pair: {positions_within_radius[-1,:]} {positions_within_radius[-1,:]}")

        return points_within_radius, positions_within_radius

    @staticmethod
    def get_grid_points_around_protein(st, grid, radius):
        point_arrays = []
        position_arrays = []
        for atom in st.protein_atoms():
            point_array, position_array = Partitioning.get_nearby_grid_points(
                grid,
                atom.pos,
                radius
            )
            point_arrays.append(point_array)
            position_arrays.append(position_array)

        all_points_array = np.concatenate(point_arrays, axis=0)
        all_positions_array = np.concatenate(position_arrays, axis=0)

        unique_points, indexes = np.unique(all_points_array, axis=0, return_index=True)
        unique_positions = all_positions_array[indexes, :]
        print(f"Unique points shape: {unique_points.shape}")
        print(f"Unique positions shape: {unique_positions.shape}")

        return unique_points, unique_positions

    @staticmethod
    def from_structure(structure: StructureInterface,
                       grid: CrystallographicGridInterface,
                       mask_radius: float,
                       mask_radius_symmetry: float,
                       debug: Debug = Debug.DEFAULT,
                       ):
        poss = []
        res_indexes = {}

        # Build a kdtree of the CA's that will be used to determine which
        # Grid points are within the vornoi cell around each CA
        for i, res_id in enumerate(structure.protein_residue_ids()):
            res_span = structure[res_id]
            res = res_span[0]
            ca = res["CA"][0]
            orthogonal = ca.pos
            poss.append(orthogonal)
            res_indexes[i] = res_id

        ca_position_array = np.array([[x for x in pos] for pos in poss])

        kdtree = spatial.KDTree(ca_position_array)

        # Determine the outer mask grid
        mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        mask.set_unit_cell(grid.unit_cell)
        for atom in structure.protein_atoms():
            pos = atom.pos
            mask.set_points_around(pos,
                                   radius=mask_radius,
                                   value=1,
                                   )
        mask_array = np.array(mask, copy=False, dtype=np.int8)

        # Get the inner mask
        inner_mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        inner_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        inner_mask.set_unit_cell(grid.unit_cell)
        for atom in structure.protein_atoms():
            pos = atom.pos
            inner_mask.set_points_around(pos,
                                         radius=mask_radius_symmetry,
                                         value=1,
                                         )
        # mask_array = np.array(mask, copy=False, dtype=np.int8)

        # Get the contact mask
        contact_mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        contact_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        contact_mask.set_unit_cell(grid.unit_cell)
        # for atom in structure.protein_atoms():
        #     pos = atom.pos
        #     contact_mask.set_points_around(pos,
        #                                    radius=4.0,
        #                                    value=1,
        #                                    )
        # mask_array = np.array(mask, copy=False, dtype=np.int8)

        # Mask the symmetry points
        symmetry_mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        symmetry_mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        symmetry_mask.set_unit_cell(grid.unit_cell)
        # symmetry_mask = Partitioning.get_symmetry_contact_mask(structure, grid, mask, mask_radius_symmetry)
        # symmetry_mask_array = np.array(symmetry_mask, copy=False, dtype=np.int8)

        # Get the grid points and their corresponding positions (NOT wrapped)
        # that are close to the protein
        point_array, position_array = Partitioning.get_grid_points_around_protein(
            structure,
            mask,
            mask_radius
        )

        # Query the position array using the kdtree to construct voronoi cells of positions
        distances, indexes = kdtree.query(position_array)

        # Construct the partitions
        partitions = {}
        for index in range(point_array.shape[0]):
            point_row = point_array[index, :]
            position_row = position_array[index, :]
            closest_ca_index = indexes[index]

            # Get the name of the residue associated with the ca
            res_id = res_indexes[closest_ca_index]

            # Begin partition if not already began
            if res_id not in partitions:
                partitions[res_id] = {}

            # Add point and position to partition
            point_tuple = (int(point_row[0]), int(point_row[1]), int(point_row[2]))
            partitions[res_id][point_tuple] = gemmi.Position(
                float(position_row[0]),
                float(position_row[1]),
                float(position_row[2])
            )

        total_mask = np.zeros(mask_array.shape, dtype=np.int8)
        total_mask[:, :, :] = mask_array[:, :, :]

        return Partitioning(
            partitions,
            mask,
            inner_mask,
            contact_mask,
            symmetry_mask,
            total_mask,
        )

    def coord_tuple(self):

        coord_array = self.coord_array()
        coord_tuple = (coord_array[:, 0],
                       coord_array[:, 1],
                       coord_array[:, 2],
                       )

        return coord_tuple

    def coord_array(self):

        coord_list = []
        for res_id in self.partitioning:
            for coord in self.partitioning[res_id]:
                coord_list.append(coord)

        return np.array(coord_list)

    @staticmethod
    def get_symmetry_contact_mask(structure: Structure, grid: gemmi.FloatGrid,
                                  protein_mask: gemmi.Int8Grid,
                                  symmetry_mask_radius: float = 3):
        protein_mask_array = np.array(protein_mask, copy=False, dtype=np.int8)

        mask = gemmi.Int8Grid(*protein_mask_array.shape)
        mask.spacegroup = protein_mask.spacegroup
        mask.set_unit_cell(protein_mask.unit_cell)

        # Mask psacegroup summetry related
        symops = Symops.from_grid(grid)

        # Symmetry waters can be a problem for known hits! See BAZ2BA-x447 for an example
        for atom in structure.all_atoms(exclude_waters=True):
            position = atom.pos
            fractional_position = mask.unit_cell.fractionalize(position)
            wrapped_position = fractional_position.wrap_to_unit()
            for symmetry_operation in symops.symops[1:]:
                symmetry_position = gemmi.Fractional(*symmetry_operation.apply_to_xyz([wrapped_position[0],
                                                                                       wrapped_position[1],
                                                                                       wrapped_position[2],
                                                                                       ]))
                orthogonal_symmetry_position = mask.unit_cell.orthogonalize(symmetry_position)

                mask.set_points_around(orthogonal_symmetry_position,
                                       radius=symmetry_mask_radius,
                                       value=1,
                                       )

        mask_array = np.array(mask, copy=False, dtype=np.int8)

        equal_mask = protein_mask_array == mask_array

        protein_mask_indicies = np.nonzero(protein_mask_array)
        protein_mask_bool = np.full(protein_mask_array.shape, False)
        protein_mask_bool[protein_mask_indicies] = True

        mask_array[~protein_mask_bool] = 0

        # ##################################
        # Mask unit cell overlap
        # ################################
        # Make a temp mask to hold overlap
        mask_unit_cell = gemmi.Int8Grid(*protein_mask_array.shape)
        mask_unit_cell.spacegroup = protein_mask.spacegroup
        mask_unit_cell.set_unit_cell(protein_mask.unit_cell)
        mask_unit_cell_array = np.array(mask_unit_cell, copy=False, dtype=np.int8)

        # Assign atoms to unit cells
        unit_cell_index_dict = {}
        for atom in structure.all_atoms():
            position = atom.pos
            fractional_position = mask_unit_cell.unit_cell.fractionalize(position)

            unit_cell_index_tuple = (int(fractional_position.x),
                                     int(fractional_position.y),
                                     int(fractional_position.z),
                                     )

            if unit_cell_index_tuple not in unit_cell_index_dict:
                unit_cell_index_dict[unit_cell_index_tuple] = []

            unit_cell_index_dict[unit_cell_index_tuple].append(position)

        # Create masks of those unit cells
        unit_cell_mask_dict = {}
        for unit_cell_index, unit_cell_position_list in unit_cell_index_dict.items():
            unit_cell_mask = gemmi.Int8Grid(*protein_mask_array.shape)
            unit_cell_mask.spacegroup = protein_mask.spacegroup
            unit_cell_mask.set_unit_cell(protein_mask.unit_cell)

            for position in unit_cell_position_list:
                mask_unit_cell.set_points_around_non_overlapping(
                    position,
                    radius=symmetry_mask_radius,
                    value=1,
                )

            unit_cell_mask_dict[unit_cell_index] = unit_cell_mask

        # Overlay those masks
        for unit_cell_index, unit_cell_mask in unit_cell_mask_dict.items():
            unit_cell_mask_array = np.array(unit_cell_mask, copy=False, dtype=np.int8)
            mask_unit_cell_array[unit_cell_mask_array == 1] += 1

        # Select the unit cell overlapping points
        mask_unit_cell_array_mask = mask_unit_cell_array > 1

        # mask them in the symmetry mask array
        mask_array[mask_unit_cell_array_mask] = 1

        return mask

    def save_maps(self, dir: Path, p1: bool = True):
        # Protein mask
        ccp4 = gemmi.Ccp4Mask()
        ccp4.grid = self.protein_mask
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(0, True)
        ccp4.write_ccp4_map(str(dir / PANDDA_PROTEIN_MASK_FILE))

        # Symmetry mask
        ccp4 = gemmi.Ccp4Mask()
        ccp4.grid = self.symmetry_mask
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(0, True)
        ccp4.write_ccp4_map(str(dir / PANDDA_SYMMETRY_MASK_FILE))

        # Total mask
        template_grid = self.symmetry_mask
        total_mask_grid = gemmi.Int8Grid(
            template_grid.nu,
            template_grid.nv,
            template_grid.nw,
        )
        total_mask_grid.spacegroup = template_grid.spacegroup
        total_mask_grid.set_unit_cell(template_grid.unit_cell)
        total_grid_array = np.array(total_mask_grid, copy=False, dtype=np.int8)
        total_grid_array[:, :, :] = self.total_mask[:, :, :]

        ccp4 = gemmi.Ccp4Mask()
        ccp4.grid = total_mask_grid
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(0, True)
        ccp4.write_ccp4_map(str(dir / PANDDA_TOTAL_MASK_FILE))

    def __getstate__(self):
        # partitioning_python = PartitoningPython.from_gemmi(self.partitioning)
        partitioning_python = self.partitioning
        protein_mask_python = Int8GridPython.from_gemmi(self.protein_mask)
        inner_mask_python = Int8GridPython.from_gemmi(self.inner_mask)
        contact_mask_python = Int8GridPython.from_gemmi(self.contact_mask)
        symmetry_mask_python = Int8GridPython.from_gemmi(self.symmetry_mask)
        return (partitioning_python,
                protein_mask_python,
                inner_mask_python,
                contact_mask_python,
                symmetry_mask_python,
                self.total_mask
                )

    def __setstate__(self, data):
        partitioning_gemmi = data[0]
        protein_mask_gemmi = data[1].to_gemmi()
        inner_mask_gemmi = data[2].to_gemmi()
        contact_mask_gemmi = data[3].to_gemmi()
        symmetry_mask_gemmi = data[4].to_gemmi()

        self.partitioning = partitioning_gemmi
        self.protein_mask = protein_mask_gemmi
        self.inner_mask = inner_mask_gemmi
        self.contact_mask = contact_mask_gemmi
        self.symmetry_mask = symmetry_mask_gemmi
        self.total_mask = data[5]


@dataclasses.dataclass()
class Grid(GridInterface):
    grid: gemmi.FloatGrid
    partitioning: Partitioning

    def new_grid(self):
        spacing = [self.grid.nu, self.grid.nv, self.grid.nw]
        unit_cell = self.grid.unit_cell
        grid = gemmi.FloatGrid(spacing[0], spacing[1], spacing[2])
        grid.set_unit_cell(unit_cell)
        grid.spacegroup = self.grid.spacegroup
        grid_array = np.array(grid, copy=False)
        grid_array[:, :, :] = 0

        return grid

    @staticmethod
    def spacing_from_reference(reference: Reference, sample_rate: float = 3.0):
        spacing = reference.dataset.reflections.reflections.get_size_for_hkl(sample_rate=sample_rate)
        return spacing

    @staticmethod
    def unit_cell_from_reference(reference: Reference):
        return reference.dataset.reflections.reflections.cell

    def __getitem__(self, item):
        return self.grid[item]

    def volume(self) -> float:
        unit_cell = self.grid.unit_cell
        return unit_cell.volume

    def size(self) -> int:
        grid = self.grid
        return grid.nu * grid.nv * grid.nw

    def shape(self):
        grid = self.grid
        return [grid.nu, grid.nv, grid.nw]

    def __getstate__(self):
        grid_python = Int8GridPython.from_gemmi(self.grid)
        partitioning_python = self.partitioning.__getstate__()

        return (grid_python, partitioning_python)

    def __setstate__(self, data):
        self.partitioning = Partitioning(data[1][0],
                                         data[1][1].to_gemmi(),
                                         data[1][2].to_gemmi(),
                                         data[1][3].to_gemmi(),
                                         data[1][4].to_gemmi(),
                                         data[1][5]
                                         )
        self.grid = data[0].to_gemmi()


@dataclasses.dataclass()
class ReferenceGrid(GridInterface):
    spacing: Tuple[int, int, int]
    unit_cell: Tuple[float, float, float, float, float, float]
    spacegroup: int
    partitioning: Partitioning

    def new_grid(self):
        # spacing = [self.grid.nu, self.grid.nv, self.grid.nw]
        # unit_cell = self.grid.unit_cell

        grid = gemmi.FloatGrid(*self.spacing)
        grid.set_unit_cell(gemmi.UnitCell(*self.unit_cell))
        grid.spacegroup = gemmi.find_spacegroup_by_number(self.spacegroup)
        grid_array = np.array(grid, copy=False)
        grid_array[:, :, :] = 0

        return grid

    @staticmethod
    def spacing_from_reference(reference: ReferenceInterface, step, offset, sample_rate: float = 3.0, ):
        step = step
        # spacing = reference.dataset.reflections.reflections.get_size_for_hkl(sample_rate=sample_rate)
        offset = offset
        structure_poss = []
        for model in reference.dataset.structure.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        pos = atom.pos
                        structure_poss.append([pos.x, pos.y, pos.z])

        pos_array = np.array(structure_poss)
        max_pos = np.max(pos_array, axis=0) + offset

        spacing_array = np.ceil(max_pos / step)

        return tuple(int(x) for x in spacing_array)

    @staticmethod
    def unit_cell_from_reference(reference: ReferenceInterface):
        offset = 3.0
        cell = reference.dataset.reflections.reflections.cell
        return (
            cell.a,
            cell.b,
            cell.c,
            cell.alpha,
            cell.beta,
            cell.gamma
        )

    # def __getitem__(self, item):
    #     return self.grid[item]

    def volume(self) -> float:
        unit_cell = gemmi.UnitCell(*self.unit_cell)
        return unit_cell.volume

    def size(self) -> int:
        spacing = self.spacing
        return spacing[0] * spacing[1] * spacing[2]

    def shape(self):
        return self.spacing

    def __getstate__(self):
        # grid_python = Int8GridPython.from_gemmi(self.grid)
        partitioning_python = self.partitioning.__getstate__()

        return (self.spacing, self.unit_cell, self.spacegroup, partitioning_python)

    def __setstate__(self, data):
        self.partitioning = Partitioning(
            data[3][0],
            data[3][1].to_gemmi(),
            data[3][2].to_gemmi(),
            data[3][3].to_gemmi(),
            data[3][4].to_gemmi(),
            data[3][5]
        )
        # self.grid = data[0].to_gemmi()
        self.spacing = data[0]
        self.unit_cell = data[1]
        self.spacegroup = data[2]


def get_grid_from_reference_dep(
        reference: ReferenceInterface,
        mask_radius: float,
        mask_radius_symmetry: float,
        sample_rate: float = 3.0,
        debug: Debug = Debug.DEFAULT
):
    unit_cell = Grid.unit_cell_from_reference(reference)
    spacing: typing.List[int] = Grid.spacing_from_reference(reference, sample_rate)

    grid = gemmi.FloatGrid(*spacing)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    grid.set_unit_cell(unit_cell)
    grid.spacegroup = reference.dataset.reflections.spacegroup()

    partitioning = Partitioning.from_reference(
        reference,
        grid,
        mask_radius,
        mask_radius_symmetry,
        debug=debug
    )

    return Grid(grid, partitioning)


def get_grid_from_reference(
        reference: ReferenceInterface,
        mask_radius: float,
        mask_radius_symmetry: float,
        sample_rate: float = 3.0,
        debug: Debug = Debug.DEFAULT
):
    # unit_cell = ReferenceGrid.unit_cell_from_reference(reference)
    step = 0.5
    offset = 6.0
    spacing = ReferenceGrid.spacing_from_reference(reference, step, offset, sample_rate, )
    unit_cell = (
        float(spacing[0] * step),
        float(spacing[1] * step),
        float(spacing[2] * step),
        float(90),
        float(90),
        float(90)
    )

    grid = gemmi.FloatGrid(*spacing)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    grid.set_unit_cell(gemmi.UnitCell(*unit_cell))
    # grid.spacegroup = reference.dataset.reflections.spacegroup()

    partitioning = Partitioning.from_reference(
        reference,
        grid,
        mask_radius,
        mask_radius_symmetry,
        debug=debug
    )

    return ReferenceGrid(spacing, unit_cell, gemmi.find_spacegroup_by_name("P 1").number, partitioning)


class GetGrid(GetGridInterface):
    def __call__(self,
                 reference: ReferenceInterface,
                 outer_mask: float,
                 inner_mask_symmetry: float,
                 sample_rate: float,
                 debug: Debug = Debug.DEFAULT,
                 ) -> GridInterface:
        return get_grid_from_reference(
            reference,
            outer_mask,
            inner_mask_symmetry,
            sample_rate,
            debug=debug
        )
