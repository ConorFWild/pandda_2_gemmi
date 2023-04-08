import itertools
import time

import gemmi
import scipy
import numpy as np

from ..interfaces import *
from ..dataset import ResidueID
from ..dmaps import SparseDMap
from ..processor import Partial


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
        # print(f"##########")

        x, y, z = position.x, position.y, position.z

        corners = []
        for dx, dy, dz in itertools.product([-radius, + radius], [-radius, + radius], [-radius, + radius]):
            corner = gemmi.Position(x + dx, y + dy, z + dz)
            corner_fractional = grid.unit_cell.fractionalize(corner)
            # corner_fractional_2 = PointPositionArray.fractionalize_orthogonal_array(
            #     np.array([corner.x, corner.y, corner.z]).reshape((1,3)),
            #     np.array(grid.unit_cell.fractionalization_matrix.tolist())
            # )
            # print(f"{(corner_fractional.x, corner_fractional.y, corner_fractional.z)} {corner_fractional_2}")
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
                # [grid.nu, grid.nv, grid.nw],
                grid
            ),
            # np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
            grid
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
        #
        # print(f"Original position was: {position.x} {position.y} {position.z}")
        # print(f"Orth bounding box min: {orth_bounds_min}")
        # print(f"Orth bounding box max: {orth_bounds_max}")
        # print(f"Point bounding box min: {point_bounds_min}")
        # print(f"Point bounding box max: {point_bounds_max}")
        # print(f"First point pos pair: {points_within_radius[0, :]} {positions_within_radius[0, :]}")
        # print(f"Last point pos pair: {points_within_radius[-1, :]} {positions_within_radius[-1, :]}")

        return points_within_radius.astype(int), positions_within_radius

    # @staticmethod
    # def get_nearby_grid_points_parallel(
    #         spacing,
    #         fractionalization_matrix,
    #         orthogonalization_matrix,
    #         position,
    #         radius,
    # ):
    #     # Get the fractional position
    #     # print(f"##########")
    #     time_begin = time.time()
    #     x, y, z = position[0], position[1], position[2]
    #
    #     corners = []
    #     for dx, dy, dz in itertools.product([-radius, + radius], [-radius, + radius], [-radius, + radius]):
    #         # corner = gemmi.Position(x + dx, y + dy, z + dz)
    #         corner_fractional = PointPositionArray.fractionalize_orthogonal_array_mat(
    #             np.array([x + dx, y + dy, z + dz]).reshape((1, 3)),
    #             fractionalization_matrix,
    #         )
    #         # print(corner_fractional.shape)
    #         corners.append([corner_fractional[0, 0], corner_fractional[0, 1], corner_fractional[0, 2]])
    #
    #     fractional_corner_array = np.array(corners)
    #     # print(fractional_corner_array.shape)
    #
    #     fractional_min = np.min(fractional_corner_array, axis=0)
    #     fractional_max = np.max(fractional_corner_array, axis=0)
    #
    #     # print(f"Fractional min: {fractional_min}")
    #     # print(f"Fractional max: {fractional_max}")
    #
    #     # Find the fractional bounding box
    #     # x, y, z = fractional.x, fractional.y, fractional.z
    #     # dx = radius / grid.nu
    #     # dy = radius / grid.nv
    #     # dz = radius / grid.nw
    #     #
    #     # # Find the grid bounding box
    #     # u0 = np.floor((x - dx) * grid.nu)
    #     # u1 = np.ceil((x + dx) * grid.nu)
    #     # v0 = np.floor((y - dy) * grid.nv)
    #     # v1 = np.ceil((y + dy) * grid.nv)
    #     # w0 = np.floor((z - dz) * grid.nw)
    #     # w1 = np.ceil((z + dz) * grid.nw)
    #     u0 = np.floor(fractional_min[0] * spacing[0])
    #     u1 = np.ceil(fractional_max[0] * spacing[0])
    #     v0 = np.floor(fractional_min[1] * spacing[1])
    #     v1 = np.ceil(fractional_max[1] * spacing[1])
    #     w0 = np.floor(fractional_min[2] * spacing[2])
    #     w1 = np.ceil(fractional_max[2] * spacing[2])
    #
    #     # print(f"Fractional bounds are: u: {u0} {u1} : v: {v0} {v1} : w: {w0} {w1}")
    #
    #     # Get the grid points
    #     time_begin_itertools = time.time()
    #     # grid_point_array = np.array(
    #     #     [
    #     #         xyz_tuple
    #     #         for xyz_tuple
    #     #         in itertools.product(
    #     #         np.arange(u0, u1 + 1),
    #     #         np.arange(v0, v1 + 1),
    #     #         np.arange(w0, w1 + 1),
    #     #     )
    #     #     ]
    #     # )
    #     grid = np.mgrid[u0:u1 + 1, v0: v1 + 1, w0:w1 + 1]
    #     grid_point_array = np.hstack([grid[_j].reshape((-1,1)) for _j in (0,1,2)])
    #     time_finish_itertools = time.time()
    #     print(f"\t\t\t\t\t\tGot grid array in {round(time_finish_itertools-time_begin_itertools, 1)}")
    #
    #     # print(f"Grid point array shape: {grid_point_array.shape}")
    #     # print(f"Grid point first element: {grid_point_array[0, :]}")
    #
    #     # Get the point positions
    #     time_begin_pointpos = time.time()
    #     position_array = PointPositionArray.orthogonalize_fractional_array_mat(
    #         PointPositionArray.fractionalize_grid_point_array_mat(
    #             grid_point_array,
    #             spacing,
    #         ),
    #         orthogonalization_matrix,
    #     )
    #     time_finish_pointpos = time.time()
    #     print(f"\t\t\t\t\t\tTransformed points to pos in {round(time_finish_pointpos - time_begin_pointpos, 1)}")
    #     # print(f"")
    #     # print(f"Grid position array shape: {position_array.shape}")
    #     # print(f"Grid position first element: {position_array[0, :]}")
    #
    #     # Get the distances to the position
    #     distance_array = np.linalg.norm(
    #         position_array - np.array(position),
    #         axis=1,
    #     )
    #     # print(f"Distance array shape: {distance_array.shape}")
    #     # print(f"Distance array first element: {distance_array[0]}")
    #
    #     # Mask the points on distances
    #     points_within_radius = grid_point_array[distance_array < radius]
    #     positions_within_radius = position_array[distance_array < radius]
    #     # print(f"Had {grid_point_array.shape} points, of which {points_within_radius.shape} within radius")
    #
    #     # Bounding box orth
    #     # orth_bounds_min = np.min(positions_within_radius, axis=0)
    #     # orth_bounds_max = np.max(positions_within_radius, axis=0)
    #     # point_bounds_min = np.min(points_within_radius, axis=0)
    #     # point_bounds_max = np.max(points_within_radius, axis=0)
    #     #
    #     # print(f"Original position was: {position.x} {position.y} {position.z}")
    #     # print(f"Orth bounding box min: {orth_bounds_min}")
    #     # print(f"Orth bounding box max: {orth_bounds_max}")
    #     # print(f"Point bounding box min: {point_bounds_min}")
    #     # print(f"Point bounding box max: {point_bounds_max}")
    #     # print(f"First point pos pair: {points_within_radius[0, :]} {positions_within_radius[0, :]}")
    #     # print(f"Last point pos pair: {points_within_radius[-1, :]} {positions_within_radius[-1, :]}")
    #
    #     time_finish = time.time()
    #     print(f"\t\t\t\t\tGot pos array of shape {positions_within_radius.shape} in {round(time_finish-time_begin, 1)}")
    #
    #     return points_within_radius.astype(int), positions_within_radius

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
        # print(f"##########")
        time_begin = time.time()
        x, y, z = position[0], position[1], position[2]

        corners = []
        for dx, dy, dz in itertools.product([-radius, + radius], [-radius, + radius], [-radius, + radius]):
            # corner = gemmi.Position(x + dx, y + dy, z + dz)
            corner_fractional = PointPositionArray.fractionalize_orthogonal_array_mat(
                np.array([x + dx, y + dy, z + dz]).reshape((1, 3)),
                fractionalization_matrix,
            )
            # print(corner_fractional.shape)
            corners.append([corner_fractional[0, 0], corner_fractional[0, 1], corner_fractional[0, 2]])

        fractional_corner_array = np.array(corners)
        # print(fractional_corner_array.shape)

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
        u0 = np.floor(fractional_min[0] * spacing[0])
        u1 = np.ceil(fractional_max[0] * spacing[0])
        v0 = np.floor(fractional_min[1] * spacing[1])
        v1 = np.ceil(fractional_max[1] * spacing[1])
        w0 = np.floor(fractional_min[2] * spacing[2])
        w1 = np.ceil(fractional_max[2] * spacing[2])

        # print(f"Fractional bounds are: u: {u0} {u1} : v: {v0} {v1} : w: {w0} {w1}")

        # Get the grid points
        time_begin_itertools = time.time()
        # grid_point_array = np.array(
        #     [
        #         xyz_tuple
        #         for xyz_tuple
        #         in itertools.product(
        #         np.arange(u0, u1 + 1),
        #         np.arange(v0, v1 + 1),
        #         np.arange(w0, w1 + 1),
        #     )
        #     ]
        # )
        grid = np.mgrid[u0:u1 + 1, v0: v1 + 1, w0:w1 + 1].astype(int)
        grid_point_array = np.hstack([grid[_j].reshape((-1, 1)) for _j in (0, 1, 2)])
        time_finish_itertools = time.time()
        print(f"\t\t\t\t\t\tGot grid array in {round(time_finish_itertools - time_begin_itertools, 1)} of shape {grid_point_array.shape}")

        # print(f"Grid point array shape: {grid_point_array.shape}")
        # print(f"Grid point first element: {grid_point_array[0, :]}")

        # Get the point positions
        time_begin_pointpos = time.time()
        mod_point_array = np.mod(grid_point_array, spacing)
        mod_point_indexes = (mod_point_array[:, 0].flatten(), mod_point_array[:, 1].flatten(), mod_point_array[:, 2].flatten())
        position_array = np.zeros(grid_point_array.shape)
        print(f"\t\t\t\t\t\tInitial position array shape: {position_array.shape}")

        position_array[:, 0] = pos_array[0][mod_point_indexes]
        position_array[:, 1] = pos_array[1][mod_point_indexes]
        position_array[:, 2] = pos_array[2][mod_point_indexes]

        # position_array = pos_array[:, , ].T

        time_finish_pointpos = time.time()
        print(f"\t\t\t\t\t\tTransformed points to pos in {round(time_finish_pointpos - time_begin_pointpos, 1)} to shape {position_array.shape}")
        # print(f"")
        # print(f"Grid position array shape: {position_array.shape}")
        # print(f"Grid position first element: {position_array[0, :]}")

        # Get the distances to the position
        distance_array = np.linalg.norm(
            position_array - np.array(position),
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
        #
        # print(f"Original position was: {position.x} {position.y} {position.z}")
        # print(f"Orth bounding box min: {orth_bounds_min}")
        # print(f"Orth bounding box max: {orth_bounds_max}")
        # print(f"Point bounding box min: {point_bounds_min}")
        # print(f"Point bounding box max: {point_bounds_max}")
        # print(f"First point pos pair: {points_within_radius[0, :]} {positions_within_radius[0, :]}")
        # print(f"Last point pos pair: {points_within_radius[-1, :]} {positions_within_radius[-1, :]}")

        time_finish = time.time()
        print(
            f"\t\t\t\t\tGot pos array of shape {positions_within_radius.shape} in {round(time_finish - time_begin, 1)}")

        return points_within_radius.astype(int), positions_within_radius

    @staticmethod
    def get_nearby_grid_points_vectorized(grid, position_array, radius):
        # Get the fractional position
        # print(f"##########")

        # x, y, z = position.x, position.y, position.z

        corners = []
        for dx, dy, dz in itertools.product([-radius, + radius], [-radius, + radius], [-radius, + radius]):
            corner = position_array + np.array([dx, dy, dz])
            corner_fractional = PointPositionArray.fractionalize_orthogonal_array(corner, grid)
            corners.append(corner_fractional)

        # Axis: Atom, coord, corner
        fractional_corner_array = np.stack([corner for corner in corners], axis=-1)
        # print(f"\t\t\t\t\tFRACTIONAL corner array shape: {fractional_corner_array.shape}")

        # print(f"Fractional min: {fractional_min}")

        fractional_min = np.min(fractional_corner_array, axis=-1)
        fractional_max = np.max(fractional_corner_array, axis=-1)
        # print(f"\t\t\t\t\tFRACTIONAL min array shape: {fractional_min.shape}")

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
        u0 = np.floor(fractional_min[:, 0] * grid.nu)
        u1 = np.ceil(fractional_max[:, 0] * grid.nu)
        v0 = np.floor(fractional_min[:, 1] * grid.nv)
        v1 = np.ceil(fractional_max[:, 1] * grid.nv)
        w0 = np.floor(fractional_min[:, 2] * grid.nw)
        w1 = np.ceil(fractional_max[:, 2] * grid.nw)

        # print(f"Fractional bounds are: u: {u0} {u1} : v: {v0} {v1} : w: {w0} {w1}")

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
            #     np.array(
            #     [
            #         xyz_tuple
            #         for xyz_tuple
            #         in itertools.product(
            #             np.arange(u0, u1 + 1),
            #             np.arange(v0, v1 + 1),
            #             np.arange(w0, w1 + 1),
            #     )
            #     ]
            # )
            # print(f"Grid point array shape: {grid_point_array.shape}")
            # print(f"Grid point first element: {grid_point_array[0, :]}")

            # Get the point positions
            points_position_array = PointPositionArray.orthogonalize_fractional_array(
                PointPositionArray.fractionalize_grid_point_array(
                    grid_point_array,
                    grid,
                ),
                grid,
            )
            # print(f"\t\t\t\t\t\tPoint position array shape: {fractional_min.shape}")

            # print(f"Grid position array shape: {position_array.shape}")
            # print(f"Grid position first element: {position_array[0, :]}")

            # Get the distances to the position
            distance_array = np.linalg.norm(
                points_position_array - position_array[j, :],
                axis=1,
            )
            # print(f"Distance array shape: {distance_array.shape}")
            # print(f"Distance array first element: {distance_array[0]}")

            # Mask the points on distances
            points_within_radius = grid_point_array[distance_array < radius]
            positions_within_radius = points_position_array[distance_array < radius]
            point_arrays.append(points_within_radius.astype(int))
            position_arrays.append(positions_within_radius)
        # print(f"Had {grid_point_array.shape} points, of which {points_within_radius.shape} within radius")

        # Bounding box orth
        # orth_bounds_min = np.min(positions_within_radius, axis=0)
        # orth_bounds_max = np.max(positions_within_radius, axis=0)
        # point_bounds_min = np.min(points_within_radius, axis=0)
        # point_bounds_max = np.max(points_within_radius, axis=0)
        #
        # print(f"Original position was: {position.x} {position.y} {position.z}")
        # print(f"Orth bounding box min: {orth_bounds_min}")
        # print(f"Orth bounding box max: {orth_bounds_max}")
        # print(f"Point bounding box min: {point_bounds_min}")
        # print(f"Point bounding box max: {point_bounds_max}")
        # print(f"First point pos pair: {points_within_radius[0, :]} {positions_within_radius[0, :]}")
        # print(f"Last point pos pair: {points_within_radius[-1, :]} {positions_within_radius[-1, :]}")

        return point_arrays, position_arrays

    @staticmethod
    def get_grid_points_around_protein(st: StructureInterface, grid, indicies, radius, processor: ProcessorInterface):
        # point_arrays = []
        # position_arrays = []
        #
        # time_begin_orth = time.time()
        # pos_array_3d = np.zeros((3, grid.nu, grid.nv, grid.nw))
        # print(np.max(pos_array_3d))
        #
        # # np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
        # point_orthogonalization_matrix = np.matmul(
        #     np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
        #     np.diag((1 / grid.nu, 1 / grid.nv, 1 / grid.nw,))
        # )
        # indicies_point_array = np.vstack(indicies)
        # print(f"\t\t\t\t\tindicies_point_array shape: {indicies_point_array.shape}")
        #
        # pos_array = np.matmul(point_orthogonalization_matrix, indicies_point_array)
        # print(f"\t\t\t\t\tPos array shape: {pos_array.shape}")
        # print(pos_array_3d[0][indicies].shape)
        # pos_array_3d[0][indicies] = pos_array[0, :]
        # pos_array_3d[1][indicies] = pos_array[1, :]
        # pos_array_3d[2][indicies] = pos_array[2, :]
        # print(np.max(pos_array_3d))
        # #
        # pos_array_3d_ref = processor.put(pos_array_3d)
        # time_finish_orth = time.time()
        # print(
        #     f"\t\t\t\tOrthogonalized mask positions in {round(time_finish_orth - time_begin_orth, 1)} to shape {pos_array.shape}")

        begin = time.time()
        positions = []

        for atom in st.protein_atoms():
            pos = atom.pos
            positions.append([pos.x, pos.y, pos.z])
            # point_array, position_array = PointPositionArray.get_nearby_grid_points(
            #     grid,
            #     atom.pos,
            #     radius
            # )
            # point_arrays.append(point_array)
            # position_arrays.append(position_array)

        pos_array = np.array(positions)

        spacing = np.array([grid.nu, grid.nv, grid.nw])
        fractionalization_matrix = np.array(grid.unit_cell.fractionalization_matrix.tolist())

        time_begin_make_array = time.time()
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

        corner_array = np.array(corners)


        fractional_min = np.min(corner_array, axis=0)
        fractional_max = np.max(corner_array, axis=0)

        u0 = np.floor(fractional_min[0] * spacing[0])
        u1 = np.ceil(fractional_max[0] * spacing[0])
        v0 = np.floor(fractional_min[1] * spacing[1])
        v1 = np.ceil(fractional_max[1] * spacing[1])
        w0 = np.floor(fractional_min[2] * spacing[2])
        w1 = np.ceil(fractional_max[2] * spacing[2])

        # print(f"Fractional bounds are: u: {u0} {u1} : v: {v0} {v1} : w: {w0} {w1}")

        # Get the grid points
        time_begin_itertools = time.time()
        # grid_point_array = np.array(
        #     [
        #         xyz_tuple
        #         for xyz_tuple
        #         in itertools.product(
        #         np.arange(u0, u1 + 1),
        #         np.arange(v0, v1 + 1),
        #         np.arange(w0, w1 + 1),
        #     )
        #     ]
        # )
        grid = np.mgrid[u0:u1 + 1, v0: v1 + 1, w0:w1 + 1].astype(int)

        # grid_point_array = np.hstack([grid[_j].reshape((-1, 1)) for _j in (0, 1, 2)])
        grid_point_indicies = [grid[_j].reshape((-1, 1)) for _j in (0, 1, 2)]

        time_finish_itertools = time.time()
        print(
            f"\t\t\t\t\t\tGot grid array in {round(time_finish_itertools - time_begin_itertools, 1)} of shape {grid_point_array.shape}")

        # print(f"Grid point array shape: {grid_point_array.shape}")
        # print(f"Grid point first element: {grid_point_array[0, :]}")

        # Get the grid points in the mask


        shifted_grid_point_indicies = [np.mod(grid_point_indicies[_j], spacing[_j]) for _j in (0,1,2)]
        mask_array = np.zeros(spacing, dtype=np.bool)
        mask_array[indicies] = True
        indicies_mask = mask_array[shifted_grid_point_indicies]

        grid_point_array = np.vstack([grid_point_indicies[_j][indicies_mask] for _j in (0,1,2)]).astype(np.int)

        point_orthogonalization_matrix = np.matmul(
            np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
            np.diag((1 / grid.nu, 1 / grid.nv, 1 / grid.nw,))
        )
        # indicies_point_array = np.vstack(indicies)

        unique_points = grid_point_array.T
        unique_positions = np.matmul(point_orthogonalization_matrix, grid_point_array).T

        time_finish_make_array = time.time()

        print(f"\t\t\t\t\t Got point and pos array in {np.round(time_finish_make_array-time_begin_make_array,1)}")
        print(f"\t\t\t\t\t With shapes {unique_points.shape} {unique_positions.shape}")

        # all_point_array = grid


        # Get the point positions
        time_begin_pointpos = time.time()
        mod_point_array = np.mod(grid_point_array, spacing)
        mod_point_indexes = (
        mod_point_array[:, 0].flatten(), mod_point_array[:, 1].flatten(), mod_point_array[:, 2].flatten())
        position_array = np.zeros(grid_point_array.shape)
        print(f"\t\t\t\t\t\tInitial position array shape: {position_array.shape}")

        position_array[:, 0] = pos_array[0][mod_point_indexes]
        position_array[:, 1] = pos_array[1][mod_point_indexes]
        position_array[:, 2] = pos_array[2][mod_point_indexes]

        # position_array = pos_array[:, , ].T

        time_finish_pointpos = time.time()
        print(
            f"\t\t\t\t\t\tTransformed points to pos in {round(time_finish_pointpos - time_begin_pointpos, 1)} to shape {position_array.shape}")

        time_finish = time.time()
        print(
            f"\t\t\t\t\tGot pos array of shape {positions_within_radius.shape} in {round(time_finish - time_begin, 1)}")

        # point_position_arrays = processor(
        #     [
        #         Partial(PointPositionArray.get_nearby_grid_points_parallel).paramaterise(
        #             [grid.nu, grid.nv, grid.nw],
        #             np.array(grid.unit_cell.fractionalization_matrix.tolist()),
        #             np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
        #             pos_array_3d_ref,
        #             atom_positions[j, :],
        #             radius,
        #         )
        #         for j
        #         in range(atom_positions.shape[0])
        #     ]
        # )



        # point_arrays, position_arrays = PointPositionArray.get_nearby_grid_points_vectorized(grid, atom_positions, radius)
        # for j in range(atom_positions.shape[0]):
        #     point_array, position_array = PointPositionArray.get_nearby_grid_points_parallel(
        #         [grid.nu, grid.nv, grid.nw],
        #         np.array(grid.unit_cell.fractionalization_matrix.tolist()),
        #         np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
        #         atom_positions[j, :],
        #         radius
        #     )
        for point_position_array in point_position_arrays:
            point_arrays.append(point_position_array[0])
            position_arrays.append(point_position_array[1])

        finish = time.time()
        print(f"\t\t\t\tGot nearby grid point position arrays in: {finish - begin}")

        _all_points_array = np.concatenate(point_arrays, axis=0)
        all_points_array = _all_points_array - np.min(_all_points_array, axis=0).reshape((1, 3))
        all_positions_array = np.concatenate(position_arrays, axis=0)

        # print(f"All points shape: {all_points_array.shape}")
        # print(f"All positions shape: {all_positions_array.shape}")

        begin = time.time()
        # unique_points, indexes = np.unique(all_points_array, axis=0, return_index=True)
        all_point_indexes = (all_points_array[:, 0], all_points_array[:, 1], all_points_array[:, 2],)
        shape = (np.max(all_points_array, axis=0) - np.min(all_points_array, axis=0)) + 1
        point_3d_array = np.zeros((shape[0], shape[1], shape[2]), dtype=bool)
        point_3d_array[all_point_indexes] = True
        initial_unique_points = np.argwhere(point_3d_array)
        unique_points = initial_unique_points + np.min(_all_points_array, axis=0).reshape((1, 3))
        unique_points_indexes = (initial_unique_points[:, 0], initial_unique_points[:, 1], initial_unique_points[:, 2],)
        pos_3d_arr_x = np.zeros((shape[0], shape[1], shape[2]))
        pos_3d_arr_y = np.zeros((shape[0], shape[1], shape[2]))
        pos_3d_arr_z = np.zeros((shape[0], shape[1], shape[2]))

        pos_3d_arr_x[all_point_indexes] = all_positions_array[:, 0]
        pos_3d_arr_y[all_point_indexes] = all_positions_array[:, 1]
        pos_3d_arr_z[all_point_indexes] = all_positions_array[:, 2]
        unique_positions = np.hstack(
            [
                pos_3d_arr_x[unique_points_indexes].reshape((-1, 1)),
                pos_3d_arr_y[unique_points_indexes].reshape((-1, 1)),
                pos_3d_arr_z[unique_points_indexes].reshape((-1, 1)),
            ]
        )

        finish = time.time()
        print(
            f"\t\t\t\tGot unique points in: {finish - begin} with point shape {unique_points.shape} and pos shape {unique_positions.shape}")

        # unique_positions = all_positions_array[indexes, :]
        # print(f"Unique points shape: {unique_points.shape}")
        # print(f"Unique positions shape: {unique_positions.shape}")

        return unique_points, unique_positions

    # @staticmethod
    # def get_grid_points_around_protein(st: StructureInterface, grid, indicies, radius, processor: ProcessorInterface):
    #     point_arrays = []
    #     position_arrays = []
    #
    #     time_begin_orth = time.time()
    #     pos_array_3d = np.zeros((3, grid.nu, grid.nv, grid.nw))
    #     print(np.max(pos_array_3d))
    #
    #     # np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
    #     point_orthogonalization_matrix = np.matmul(
    #         np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
    #         np.diag((1 / grid.nu, 1 / grid.nv, 1 / grid.nw,))
    #     )
    #     indicies_point_array = np.vstack(indicies)
    #     print(f"\t\t\t\t\tindicies_point_array shape: {indicies_point_array.shape}")
    #
    #     pos_array = np.matmul(point_orthogonalization_matrix, indicies_point_array)
    #     print(f"\t\t\t\t\tPos array shape: {pos_array.shape}")
    #     print(pos_array_3d[0][indicies].shape)
    #     pos_array_3d[0][indicies] = pos_array[0, :]
    #     pos_array_3d[1][indicies] = pos_array[1, :]
    #     pos_array_3d[2][indicies] = pos_array[2, :]
    #     print(np.max(pos_array_3d))
    #     #
    #     pos_array_3d_ref = processor.put(pos_array_3d)
    #     time_finish_orth = time.time()
    #     print(f"\t\t\t\tOrthogonalized mask positions in {round(time_finish_orth-time_begin_orth, 1)} to shape {pos_array.shape}")
    #
    #     begin = time.time()
    #     positions = []
    #
    #     for atom in st.protein_atoms():
    #         pos = atom.pos
    #         positions.append([pos.x, pos.y, pos.z])
    #         # point_array, position_array = PointPositionArray.get_nearby_grid_points(
    #         #     grid,
    #         #     atom.pos,
    #         #     radius
    #         # )
    #         # point_arrays.append(point_array)
    #         # position_arrays.append(position_array)
    #
    #     atom_positions = np.array(positions)
    #
    #     point_position_arrays = processor(
    #         [
    #             Partial(PointPositionArray.get_nearby_grid_points_parallel).paramaterise(
    #                 [grid.nu, grid.nv, grid.nw],
    #                 np.array(grid.unit_cell.fractionalization_matrix.tolist()),
    #                 np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
    #                 pos_array_3d_ref,
    #                 atom_positions[j, :],
    #                 radius,
    #             )
    #             for j
    #             in range(atom_positions.shape[0])
    #         ]
    #     )
    #
    #     # point_arrays, position_arrays = PointPositionArray.get_nearby_grid_points_vectorized(grid, atom_positions, radius)
    #     # for j in range(atom_positions.shape[0]):
    #     #     point_array, position_array = PointPositionArray.get_nearby_grid_points_parallel(
    #     #         [grid.nu, grid.nv, grid.nw],
    #     #         np.array(grid.unit_cell.fractionalization_matrix.tolist()),
    #     #         np.array(grid.unit_cell.orthogonalization_matrix.tolist()),
    #     #         atom_positions[j, :],
    #     #         radius
    #     #     )
    #     for point_position_array in point_position_arrays:
    #         point_arrays.append(point_position_array[0])
    #         position_arrays.append(point_position_array[1])
    #
    #     finish = time.time()
    #     print(f"\t\t\t\tGot nearby grid point position arrays in: {finish - begin}")
    #
    #     _all_points_array = np.concatenate(point_arrays, axis=0)
    #     all_points_array = _all_points_array - np.min(_all_points_array, axis=0).reshape((1, 3))
    #     all_positions_array = np.concatenate(position_arrays, axis=0)
    #
    #     # print(f"All points shape: {all_points_array.shape}")
    #     # print(f"All positions shape: {all_positions_array.shape}")
    #
    #     begin = time.time()
    #     # unique_points, indexes = np.unique(all_points_array, axis=0, return_index=True)
    #     all_point_indexes = (all_points_array[:, 0], all_points_array[:, 1], all_points_array[:, 2],)
    #     shape = (np.max(all_points_array, axis=0) - np.min(all_points_array, axis=0)) + 1
    #     point_3d_array = np.zeros((shape[0], shape[1], shape[2]), dtype=bool)
    #     point_3d_array[all_point_indexes] = True
    #     initial_unique_points = np.argwhere(point_3d_array)
    #     unique_points = initial_unique_points + np.min(_all_points_array, axis=0).reshape((1, 3))
    #     unique_points_indexes = (initial_unique_points[:, 0], initial_unique_points[:, 1], initial_unique_points[:, 2],)
    #     pos_3d_arr_x = np.zeros((shape[0], shape[1], shape[2]))
    #     pos_3d_arr_y = np.zeros((shape[0], shape[1], shape[2]))
    #     pos_3d_arr_z = np.zeros((shape[0], shape[1], shape[2]))
    #
    #     pos_3d_arr_x[all_point_indexes] = all_positions_array[:, 0]
    #     pos_3d_arr_y[all_point_indexes] = all_positions_array[:, 1]
    #     pos_3d_arr_z[all_point_indexes] = all_positions_array[:, 2]
    #     unique_positions = np.hstack(
    #         [
    #             pos_3d_arr_x[unique_points_indexes].reshape((-1, 1)),
    #             pos_3d_arr_y[unique_points_indexes].reshape((-1, 1)),
    #             pos_3d_arr_z[unique_points_indexes].reshape((-1, 1)),
    #         ]
    #     )
    #
    #     finish = time.time()
    #     print(
    #         f"\t\t\t\tGot unique points in: {finish - begin} with point shape {unique_points.shape} and pos shape {unique_positions.shape}")
    #
    #     # unique_positions = all_positions_array[indexes, :]
    #     # print(f"Unique points shape: {unique_points.shape}")
    #     # print(f"Unique positions shape: {unique_positions.shape}")
    #
    #     return unique_points, unique_positions

    @classmethod
    def from_structure(cls, st: StructureInterface, grid, indicies, processor, radius: float = 6.0):
        point_array, position_array = PointPositionArray.get_grid_points_around_protein(st, grid, indicies, radius, processor)
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
        for model in structure.structure:
            for chain in model:
                for residue in chain.first_conformer():
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
            self.positions[mask, :]
        )


def contains(string, pattern):
    if pattern in string:
        return True
    else:
        return False


class GridPartitioning(GridPartitioningInterface):
    def __init__(self, dataset, grid, indicies, processor):
        # Get the structure array
        st_array = StructureArray.from_structure(dataset.structure)
        print(f"Structure array shape: {st_array.positions.shape}")

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

        begin = time.time()
        ca_point_position_array = st_array.mask(np.array(ca_mask))
        finish = time.time()
        print(f"\t\t\tGot position array in : {finish - begin}")
        print(f"\t\t\tCA array shape: {ca_point_position_array.positions.shape}")

        # Get the tree
        begin = time.time()
        kdtree = scipy.spatial.KDTree(ca_point_position_array.positions)
        finish = time.time()
        print(f"\t\t\tBuilt tree in : {finish - begin}")

        # Get the point array
        begin = time.time()
        point_position_array = PointPositionArray.from_structure(dataset.structure, grid, indicies, processor)
        finish = time.time()
        print(f"\t\t\tGot point position array : {finish - begin}")

        # Get the NN indexes
        begin = time.time()
        distances, indexes = kdtree.query(point_position_array.positions, workers=12)
        finish = time.time()
        print(f"\t\t\tQueryed points in : {finish - begin}")

        # Get partions
        self.partitions = {
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
        }


class GridMask(GridMaskInterface):
    def __init__(self, dataset: DatasetInterface, grid, mask_radius=6.0, mask_radius_inner=2.0):
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
        self.indicies_inner = np.nonzero(mask_array)
        self.indicies_sparse_inner = mask_array[self.indicies] == 1.0

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
        self.indicies_inner_atomic = np.nonzero(mask_array)
        self.indicies_sparse_inner_atomic = mask_array[self.indicies] == 1.0


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

        # Get the mask
        begin_mask = time.time()
        self.mask = GridMask(dataset, grid)
        finish_mask = time.time()
        print(f"\tGot mask in {finish_mask - begin_mask}")

        # Get the grid partitioning
        begin_partition = time.time()
        self.partitioning = GridPartitioning(dataset, grid, self.mask.indicies, processor)
        finish_partition = time.time()
        print(f"\tGot Partitions in {finish_partition - begin_partition}")

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
