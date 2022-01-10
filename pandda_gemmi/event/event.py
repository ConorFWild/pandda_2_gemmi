from __future__ import annotations

import time
import typing
from typing import *
import dataclasses

from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import DBSCAN
from joblib.externals.loky import set_loky_pickler

set_loky_pickler('pickle')

# from pandda_gemmi.pandda_functions import save_event_map
from pandda_gemmi.python_types import *
from pandda_gemmi.common import EventIDX, EventID, SiteID, Dtag, PositionsArray, delayed
from pandda_gemmi.dataset import Reference, Dataset, StructureFactors
from pandda_gemmi.edalignment import Grid, Xmap, Alignment, Xmaps, Partitioning
from pandda_gemmi.model import Zmap, Zmaps, Model
from pandda_gemmi.sites import Sites


def save_event_map(
        path,
        xmap: Xmap,
        model: Model,
        event: Event,
        dataset: Dataset,
        alignment: Alignment,
        grid: Grid,
        structure_factors: StructureFactors,
        mask_radius: float,
        mask_radius_symmetry: float,
        partitioning: Partitioning,
        sample_rate: float,
        # native_grid,
):
    reference_xmap_grid = xmap.xmap
    reference_xmap_grid_array = np.array(reference_xmap_grid, copy=True)

    # moving_xmap_grid: gemmi.FloatGrid = dataset.reflections.reflections.transform_f_phi_to_map(structure_factors.f,
    #                                                                                          structure_factors.phi,
    #                                                                                          )

    event_map_reference_grid = gemmi.FloatGrid(*[reference_xmap_grid.nu,
                                                 reference_xmap_grid.nv,
                                                 reference_xmap_grid.nw,
                                                 ]
                                               )
    event_map_reference_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")  # xmap.xmap.spacegroup
    event_map_reference_grid.set_unit_cell(reference_xmap_grid.unit_cell)

    event_map_reference_grid_array = np.array(event_map_reference_grid,
                                              copy=False,
                                              )

    mean_array = model.mean
    event_map_reference_grid_array[:, :, :] = (reference_xmap_grid_array - (event.bdc.bdc * mean_array)) / (
            1 - event.bdc.bdc)

    event_map_grid = Xmap.from_aligned_map_c(
        event_map_reference_grid,
        dataset,
        alignment,
        grid,
        structure_factors,
        mask_radius,
        partitioning,
        mask_radius_symmetry,
        sample_rate*2, #TODO: remove?
    )

    # # # Get the event bounding box
    # # Find the min and max positions
    # min_array = np.array(event.native_positions[0])
    # max_array = np.array(event.native_positions[0])
    # for position in event.native_positions:
    #     position_array = np.array(position)
    #     min_array = np.min(np.vstack(min_array, position_array), axis=0)
    #     max_array = np.max(np.vstack(max_array, position_array), axis=0)
    #
    #
    # # Get them as fractional bounding box
    # print(min_array)
    # print(max_array)
    # print(event.native_positions[0])
    # print(event.native_centroid)
    # print(event.cluster.centroid)
    #
    # box = gemmi.FractionalBox()
    # box.minimum = gemmi.Fractional(min_array[0], min_array[1], min_array[2])
    # box.maximum = gemmi.Fractional(max_array[0], max_array[1], max_array[2])

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = event_map_grid.xmap
    ccp4.update_ccp4_header(2, True)
    ccp4.setup()
    # ccp4.set_extent(box)
    # ccp4.grid.symmetrize_max()
    ccp4.write_ccp4_map(str(path))


@dataclasses.dataclass()
class Cluster:
    indexes: typing.Tuple[np.ndarray]
    cluster_positions_array: np.array
    values: np.ndarray
    centroid: Tuple[float, float, float]
    event_mask_indicies: Optional[np.ndarray]
    cluster_inner_protein_mask: np.array
    cluster_contact_mask: np.array
    time_event_mask: Optional[float] = 0.0

    def size(self, grid: Grid):
        grid_volume = grid.volume()
        grid_size = grid.size()
        grid_voxel_volume = grid_volume / grid_size
        return self.values.size * grid_voxel_volume

    def peak(self):
        return np.max(self.values)


def get_event_mask_indicies(zmap: Zmap, cluster_positions_array: np.ndarray):
    # cluster_positions_array = extrema_cart_coords_array[cluster_indicies]
    positions = PositionsArray(cluster_positions_array).to_positions()
    event_mask = gemmi.Int8Grid(*zmap.shape())
    event_mask.spacegroup = zmap.spacegroup()
    event_mask.set_unit_cell(zmap.unit_cell())
    for position in positions:
        event_mask.set_points_around(position,
                                     radius=2.0,
                                     value=1,
                                     )

    # event_mask.symmetrize_max()

    event_mask_array = np.array(event_mask, copy=True, dtype=np.int8)
    event_mask_indicies = np.nonzero(event_mask_array)
    return event_mask_indicies


@dataclasses.dataclass()
class Clustering:
    clustering: typing.Dict[int, Cluster]
    time_cluster: Optional[float] = 0.0
    time_np: Optional[float] = 0.0
    time_event_masking: Optional[float] = 0.0
    time_get_orth: Optional[float] = 0.0
    time_fcluster: Optional[float] = 0.0

    @staticmethod
    def from_zmap(zmap: Zmap, reference: Reference, grid: Grid, contour_level: float,
                  cluster_cutoff_distance_multiplier: float = 1.3):
        time_cluster_start = time.time()

        time_np_start = time.time()

        zmap_array = zmap.to_array(copy=True)

        # Get the protein mask
        protein_mask_grid = grid.partitioning.protein_mask
        protein_mask = np.array(protein_mask_grid, copy=False, dtype=np.int8)

        # Get the symmetry mask
        symmetry_contact_mask_grid = grid.partitioning.symmetry_mask
        symmetry_contact_mask = np.array(symmetry_contact_mask_grid, copy=False, dtype=np.int8)

        # Get the protein inner mask for determining which cluster points can be associated with already modelled
        # features
        inner_mask_grid = grid.partitioning.inner_mask
        inner_mask = np.array(inner_mask_grid, copy=False, dtype=np.int8)

        # Get the contact mask for determining how much could be binding
        contact_mask_grid = grid.partitioning.contact_mask
        contact_mask = np.array(contact_mask_grid, copy=False, dtype=np.int8)

        # Don't consider outlying points away from the protein
        protein_mask_bool = np.full(protein_mask.shape, False)
        protein_mask_bool[np.nonzero(protein_mask)] = True
        zmap_array[~protein_mask_bool] = 0.0

        # Don't consider outlying points at symmetry contacts
        zmap_array[np.nonzero(symmetry_contact_mask)] = 0.0

        extrema_mask_array = zmap_array > contour_level

        # Get the unwrapped coords, and convert them to unwrapped fractional positions, then orthogonal points for clustering
        point_array = grid.partitioning.coord_array()
        point_tuple = (point_array[:, 0],
                       point_array[:, 1],
                       point_array[:, 2],
                       )
        point_tuple_wrapped = (
            np.mod(point_array[:, 0], grid.grid.nu),
            np.mod(point_array[:, 1], grid.grid.nv),
            np.mod(point_array[:, 2], grid.grid.nw),
        )

        extrema_point_mask = extrema_mask_array[point_tuple_wrapped] == 1
        extrema_point_array = point_array[extrema_point_mask]
        extrema_point_wrapped_tuple = (
            point_tuple_wrapped[0][extrema_point_mask],
            point_tuple_wrapped[1][extrema_point_mask],
            point_tuple_wrapped[2][extrema_point_mask],
        )
        extrema_fractional_array = extrema_point_array / np.array([grid.grid.nu, grid.grid.nv, grid.grid.nw]).reshape(
            (1, 3))

        # TODO: possible bottleneck
        time_get_orth_pos_start = time.time()
        positions_orthogonal = [zmap.zmap.unit_cell.orthogonalize(gemmi.Fractional(fractional[0],
                                                                                   fractional[1],
                                                                                   fractional[2],
                                                                                   )) for fractional in
                                extrema_fractional_array]
        time_get_orth_pos_finish = time.time()

        positions = [[position.x, position.y, position.z] for position in positions_orthogonal]

        # positions = []
        # for point in extrema_grid_coords_array:
        #     # position = gemmi.Fractional(*point)
        #     point = grid.grid.get_point(*point)
        #     # fractional = grid.grid.point_to_fractional(point)
        #     # pos_orth = zmap.zmap.unit_cell.orthogonalize(fractional)
        #     orthogonal = grid.grid.point_to_position(point)

        #     pos_orth_array = [orthogonal[0],
        #                       orthogonal[1],
        #                       orthogonal[2], ]
        #     positions.append(pos_orth_array)

        extrema_cart_coords_array = np.array(positions)  # n, 3

        point_000 = grid.grid.get_point(0, 0, 0)
        point_111 = grid.grid.get_point(1, 1, 1)
        position_000 = grid.grid.point_to_position(point_000)
        position_111 = grid.grid.point_to_position(point_111)
        clustering_cutoff = position_000.dist(position_111) * cluster_cutoff_distance_multiplier

        if extrema_cart_coords_array.size < 10:
            clusters = {}
            return Clustering(clusters)

        time_np_finish = time.time()

        # TODO: possible bottleneck
        time_fcluster_start = time.time()
        cluster_ids_array = fclusterdata(X=extrema_cart_coords_array,
                                         # t=blob_finding.clustering_cutoff,
                                         t=clustering_cutoff,
                                         criterion='distance',
                                         metric='euclidean',
                                         method='single',
                                         )
        time_fcluster_finish = time.time()

        clusters = {}
        time_event_masking_start = time.time()
        for unique_cluster in np.unique(cluster_ids_array):
            if unique_cluster == -1:
                continue
            cluster_mask = cluster_ids_array == unique_cluster  # n
            cluster_indicies = np.nonzero(cluster_mask)  # (n')
            # cluster_points_array = extrema_point_array[cluster_indicies]
            # cluster_points_tuple = (cluster_points_array[:, 0],
            #                         cluster_points_array[:, 1],
            #                         cluster_points_array[:, 2],)

            cluster_points_tuple = (
                extrema_point_wrapped_tuple[0][cluster_indicies],
                extrema_point_wrapped_tuple[1][cluster_indicies],
                extrema_point_wrapped_tuple[2][cluster_indicies],
            )

            # Get the values of the z map at the cluster points
            values = zmap_array[cluster_points_tuple]

            # Get the inner protein mask applied to the cluster
            cluster_inner_protein_mask = inner_mask[cluster_points_tuple]

            # Generate event mask
            cluster_positions_array = extrema_cart_coords_array[cluster_indicies]

            # Generate the contact mask
            cluster_contact_mask = contact_mask[cluster_points_tuple]

            ###

            time_event_mask_start = time.time()
            # positions = PositionsArray(cluster_positions_array).to_positions()
            # event_mask = gemmi.Int8Grid(*zmap.shape())
            # event_mask.spacegroup = zmap.spacegroup()
            # event_mask.set_unit_cell(zmap.unit_cell())
            # for position in positions:
            #     event_mask.set_points_around(position,
            #                                  radius=3.0,
            #                                  value=1,
            #                                  )

            # event_mask.symmetrize_max()

            # event_mask_array = np.array(event_mask, copy=True, dtype=np.int8)
            # event_mask_indicies = np.nonzero(event_mask_array)

            time_event_mask_finish = time.time()
            ###

            centroid_array = np.mean(cluster_positions_array,
                                     axis=0)
            centroid = (centroid_array[0],
                        centroid_array[1],
                        centroid_array[2],)

            cluster = Cluster(cluster_points_tuple,
                              cluster_positions_array,
                              values,
                              centroid,
                              None,
                              cluster_inner_protein_mask,
                              cluster_contact_mask,
                              # time_get_orth=time_get_orth_pos_finish-time_get_orth_pos_start,
                              # time_fcluster=time_fcluster_finish-time_fcluster_start,
                              time_event_mask=time_event_mask_finish - time_event_mask_start,
                              )
            clusters[unique_cluster] = cluster
        time_event_masking_finish = time.time()

        time_cluster_finish = time.time()

        return Clustering(clusters,
                          time_cluster=time_cluster_finish - time_cluster_start,
                          time_np=time_np_finish - time_np_start,
                          time_event_masking=time_event_masking_finish - time_event_masking_start,
                          time_get_orth=time_get_orth_pos_finish - time_get_orth_pos_start,
                          time_fcluster=time_fcluster_finish - time_fcluster_start,
                          )

    @staticmethod
    def from_event_map(
            event_map_array,
            zmap: Zmap,
            reference: Reference,
            grid: Grid,
            contour_level: float,
            cluster_cutoff_distance_multiplier: float = 1.3,
    ):
        time_cluster_start = time.time()

        time_np_start = time.time()

        zmap_array = zmap.to_array(copy=True)
        # event_map_array = np.array(event_map, copy=True)

        # Get the protein mask
        protein_mask_grid = grid.partitioning.protein_mask
        protein_mask = np.array(protein_mask_grid, copy=False, dtype=np.int8)

        # Get the symmetry mask
        symmetry_contact_mask_grid = grid.partitioning.symmetry_mask
        symmetry_contact_mask = np.array(symmetry_contact_mask_grid, copy=False, dtype=np.int8)

        # Get the protein inner mask for determining which cluster points can be associated with already modelled
        # features
        inner_mask_grid = grid.partitioning.inner_mask
        inner_mask = np.array(inner_mask_grid, copy=False, dtype=np.int8)

        # Get the contact mask for determining how much could be binding
        contact_mask_grid = grid.partitioning.contact_mask
        contact_mask = np.array(contact_mask_grid, copy=False, dtype=np.int8)

        # Don't consider outlying points away from the protein
        protein_mask_bool = np.full(protein_mask.shape, False)
        protein_mask_bool[np.nonzero(protein_mask)] = True
        zmap_array[~protein_mask_bool] = 0.0

        # Don't consider outlying points at symmetry contacts
        zmap_array[np.nonzero(symmetry_contact_mask)] = 0.0


        # Don't consider the backbone or sidechains of the event mask unless they have a high z value
        event_map_array[~protein_mask_bool] = 0.0
        event_map_array[np.nonzero(symmetry_contact_mask)] = 0.0
        inner_mask_array = np.array(grid.partitioning.inner_mask, copy=True)
        inner_mask_array[zmap_array > contour_level] = 0.0
        event_map_array[np.nonzero(inner_mask_array)] = 0.0




        # Get the unwrapped coords, and convert them to unwrapped fractional positions, then orthogonal points for clustering
        extrema_mask_array = event_map_array > contour_level

        point_array = grid.partitioning.coord_array()
        point_tuple = (point_array[:, 0],
                       point_array[:, 1],
                       point_array[:, 2],
                       )
        point_tuple_wrapped = (
            np.mod(point_array[:, 0], grid.grid.nu),
            np.mod(point_array[:, 1], grid.grid.nv),
            np.mod(point_array[:, 2], grid.grid.nw),
        )

        extrema_point_mask = extrema_mask_array[point_tuple_wrapped] == 1
        extrema_point_array = point_array[extrema_point_mask]
        extrema_point_wrapped_tuple = (
            point_tuple_wrapped[0][extrema_point_mask],
            point_tuple_wrapped[1][extrema_point_mask],
            point_tuple_wrapped[2][extrema_point_mask],
        )
        extrema_fractional_array = extrema_point_array / np.array([grid.grid.nu, grid.grid.nv, grid.grid.nw]).reshape(
            (1, 3))

        # TODO: possible bottleneck
        time_get_orth_pos_start = time.time()
        positions_orthogonal = [zmap.zmap.unit_cell.orthogonalize(gemmi.Fractional(fractional[0],
                                                                                   fractional[1],
                                                                                   fractional[2],
                                                                                   )) for fractional in
                                extrema_fractional_array]
        time_get_orth_pos_finish = time.time()

        positions = [[position.x, position.y, position.z] for position in positions_orthogonal]

        # positions = []
        # for point in extrema_grid_coords_array:
        #     # position = gemmi.Fractional(*point)
        #     point = grid.grid.get_point(*point)
        #     # fractional = grid.grid.point_to_fractional(point)
        #     # pos_orth = zmap.zmap.unit_cell.orthogonalize(fractional)
        #     orthogonal = grid.grid.point_to_position(point)

        #     pos_orth_array = [orthogonal[0],
        #                       orthogonal[1],
        #                       orthogonal[2], ]
        #     positions.append(pos_orth_array)

        extrema_cart_coords_array = np.array(positions)  # n, 3

        point_000 = grid.grid.get_point(0, 0, 0)
        point_111 = grid.grid.get_point(1, 1, 1)
        position_000 = grid.grid.point_to_position(point_000)
        position_111 = grid.grid.point_to_position(point_111)
        clustering_cutoff = position_000.dist(position_111) * cluster_cutoff_distance_multiplier

        if extrema_cart_coords_array.size < 10:
            clusters = {}
            return Clustering(clusters)

        time_np_finish = time.time()

        # TODO: possible bottleneck
        time_fcluster_start = time.time()
        cluster_ids_array = fclusterdata(X=extrema_cart_coords_array,
                                         # t=blob_finding.clustering_cutoff,
                                         t=clustering_cutoff,
                                         criterion='distance',
                                         metric='euclidean',
                                         method='single',
                                         )
        time_fcluster_finish = time.time()

        clusters = {}
        time_event_masking_start = time.time()
        for unique_cluster in np.unique(cluster_ids_array):
            if unique_cluster == -1:
                continue
            cluster_mask = cluster_ids_array == unique_cluster  # n
            cluster_indicies = np.nonzero(cluster_mask)  # (n')
            # cluster_points_array = extrema_point_array[cluster_indicies]
            # cluster_points_tuple = (cluster_points_array[:, 0],
            #                         cluster_points_array[:, 1],
            #                         cluster_points_array[:, 2],)

            cluster_points_tuple = (
                extrema_point_wrapped_tuple[0][cluster_indicies],
                extrema_point_wrapped_tuple[1][cluster_indicies],
                extrema_point_wrapped_tuple[2][cluster_indicies],
            )

            # Get the values of the z map at the cluster points
            values = event_map_array[cluster_points_tuple]

            # Get the inner protein mask applied to the cluster
            cluster_inner_protein_mask = inner_mask[cluster_points_tuple]

            # Generate event mask
            cluster_positions_array = extrema_cart_coords_array[cluster_indicies]

            # Generate the contact mask
            cluster_contact_mask = contact_mask[cluster_points_tuple]

            ###

            time_event_mask_start = time.time()
            # positions = PositionsArray(cluster_positions_array).to_positions()
            # event_mask = gemmi.Int8Grid(*zmap.shape())
            # event_mask.spacegroup = zmap.spacegroup()
            # event_mask.set_unit_cell(zmap.unit_cell())
            # for position in positions:
            #     event_mask.set_points_around(position,
            #                                  radius=3.0,
            #                                  value=1,
            #                                  )

            # event_mask.symmetrize_max()

            # event_mask_array = np.array(event_mask, copy=True, dtype=np.int8)
            # event_mask_indicies = np.nonzero(event_mask_array)

            time_event_mask_finish = time.time()
            ###

            centroid_array = np.mean(cluster_positions_array,
                                     axis=0)
            centroid = (centroid_array[0],
                        centroid_array[1],
                        centroid_array[2],)

            cluster = Cluster(cluster_points_tuple,
                              cluster_positions_array,
                              values,
                              centroid,
                              None,
                              cluster_inner_protein_mask,
                              cluster_contact_mask,
                              # time_get_orth=time_get_orth_pos_finish-time_get_orth_pos_start,
                              # time_fcluster=time_fcluster_finish-time_fcluster_start,
                              time_event_mask=time_event_mask_finish - time_event_mask_start,
                              )
            clusters[unique_cluster] = cluster
        time_event_masking_finish = time.time()

        time_cluster_finish = time.time()

        return Clustering(clusters,
                          time_cluster=time_cluster_finish - time_cluster_start,
                          time_np=time_np_finish - time_np_start,
                          time_event_masking=time_event_masking_finish - time_event_masking_start,
                          time_get_orth=time_get_orth_pos_finish - time_get_orth_pos_start,
                          time_fcluster=time_fcluster_finish - time_fcluster_start,
                          )

    def __iter__(self):
        for cluster_num in self.clustering:
            yield cluster_num

    def cluster_mask(self, grid: Grid):
        grid_array = np.array(grid.grid, copy=False)
        mask = gemmi.Int8Grid(*grid_array.shape)
        mask.spacegroup = grid.grid.spacegroup
        mask.set_unit_cell(grid.grid.unit_cell)

        mask_array = np.array(mask, copy=False)

        for cluster_id in self.clustering:
            cluster = self.clustering[cluster_id]
            indexes = cluster.indexes
            mask_array[indexes] = 1

        return mask

    @staticmethod
    def get_protein_mask(zmap: Zmap, reference: Reference, masks_radius: float):
        mask = gemmi.Int8Grid(*zmap.shape())
        mask.spacegroup = zmap.spacegroup()
        mask.set_unit_cell(zmap.unit_cell())

        for atom in reference.dataset.structure.protein_atoms():
            pos = atom.pos
            mask.set_points_around(pos,
                                   radius=masks_radius,
                                   value=1,
                                   )

        mask_array = np.array(mask, copy=False)

        return mask

    # @staticmethod
    # def get_symmetry_contact_mask(zmap: Zmap, reference: Reference, protein_mask: gemmi.Int8Grid,
    #                               symmetry_mask_radius: float = 3):
    #     mask = gemmi.Int8Grid(*zmap.shape())
    #     mask.spacegroup = zmap.spacegroup()
    #     mask.set_unit_cell(zmap.unit_cell())
    #
    #     symops = Symops.from_grid(mask)
    #
    #     for atom in reference.dataset.structure.protein_atoms():
    #         for symmetry_operation in symops.symops[1:]:
    #             position = atom.pos
    #             fractional_position = mask.unit_cell.fractionalize(position)
    #             symmetry_position = gemmi.Fractional(*symmetry_operation.apply_to_xyz([fractional_position[0],
    #                                                                                    fractional_position[1],
    #                                                                                    fractional_position[2],
    #                                                                                    ]))
    #             orthogonal_symmetry_position = mask.unit_cell.orthogonalize(symmetry_position)
    #
    #             mask.set_points_around(orthogonal_symmetry_position,
    #                                    radius=symmetry_mask_radius,
    #                                    value=1,
    #                                    )
    #
    #     mask_array = np.array(mask, copy=False, dtype=np.int8)
    #
    #     protein_mask_array = np.array(protein_mask, copy=False, dtype=np.int8)
    #
    #     equal_mask = protein_mask_array == mask_array
    #
    #     protein_mask_indicies = np.nonzero(protein_mask_array)
    #     protein_mask_bool = np.full(protein_mask_array.shape, False)
    #     protein_mask_bool[protein_mask_indicies] = True
    #
    #     mask_array[~protein_mask_bool] = 0
    #
    #     return mask

    def __getitem__(self, item):
        return self.clustering[item]

    def __len__(self):
        return len(self.clustering)


@dataclasses.dataclass()
class Clusterings:
    clusterings: typing.Dict[Dtag, Clustering]

    @staticmethod
    def from_Zmaps(zmaps: Zmaps, reference: Reference, grid: Grid, contour_level: float,
                   cluster_cutoff_distance_multiplier: float,
                   mapper=False):

        if mapper:
            keys = list(zmaps.zmaps.keys())

            results = mapper(
                delayed(
                    Clustering.from_zmap)(
                    zmaps[key],
                    reference,
                    grid,
                    contour_level,
                    cluster_cutoff_distance_multiplier,
                )
                for key
                in keys
            )
            clusterings = {keys[i]: results[i]
                           for i, key
                           in enumerate(keys)
                           }
        else:

            clusterings = {}
            for dtag in zmaps:
                clustering = Clustering.from_zmap(zmaps[dtag], reference, grid, contour_level)
                clusterings[dtag] = clustering

        return Clusterings(clusterings)

    def filter_size(self, grid: Grid, min_cluster_size: float):
        new_clusterings = {}
        for dtag in self.clusterings:
            clustering = self.clusterings[dtag]
            new_cluster_nums = list(filter(lambda cluster_num: clustering[cluster_num].size(grid) > min_cluster_size,
                                           clustering,
                                           )
                                    )

            if len(new_cluster_nums) == 0:
                continue

            else:
                new_clusters_dict = {new_cluster_num: clustering[new_cluster_num] for new_cluster_num in
                                     new_cluster_nums}
                new_clustering = Clustering(new_clusters_dict)
                new_clusterings[dtag] = new_clustering

        return Clusterings(new_clusterings)

    def filter_peak(self, grid: Grid, z_peak: float):
        new_clusterings = {}
        for dtag in self.clusterings:
            clustering = self.clusterings[dtag]
            new_cluster_nums = list(filter(lambda cluster_num: clustering[cluster_num].peak() > z_peak,
                                           clustering,
                                           )
                                    )

            if len(new_cluster_nums) == 0:
                continue

            else:
                new_clusters_dict = {new_cluster_num: clustering[new_cluster_num] for new_cluster_num in
                                     new_cluster_nums}
                new_clustering = Clustering(new_clusters_dict)
                new_clusterings[dtag] = new_clustering

        return Clusterings(new_clusterings)

    def merge_clusters(self):
        new_clustering_dict = {}
        for dtag in self.clusterings:
            clustering = self.clusterings[dtag]

            cluster_list = []
            centroid_list = []
            for cluster_id in clustering:
                cluster = clustering[cluster_id]
                centroid = cluster.centroid
                cluster_list.append(cluster)
                centroid_list.append(centroid)

            cluster_array = np.array(cluster_list)
            centroid_array = np.array(centroid_list)

            dbscan = DBSCAN(
                eps=6,
                min_samples=1,
            )

            cluster_ids_array = dbscan.fit_predict(centroid_array)
            new_clusters = {}
            for unique_cluster in np.unique(cluster_ids_array):
                current_clusters = cluster_array[cluster_ids_array == unique_cluster]

                cluster_points_tuple = tuple(np.concatenate([current_cluster.indexes[i]
                                                             for current_cluster
                                                             in current_clusters
                                                             ], axis=None,
                                                            )
                                             for i
                                             in [0, 1, 2]
                                             )
                cluster_positions_array = np.vstack([current_cluster.cluster_positions_array
                                                     for current_cluster
                                                     in current_clusters])

                cluster_positions_list = [current_cluster.centroid
                                          for current_cluster
                                          in current_clusters
                                          ]

                values = np.concatenate([current_cluster.values
                                         for current_cluster
                                         in current_clusters
                                         ], axis=None,
                                        )

                cluster_inner_protein_mask = np.concatenate([current_cluster.cluster_inner_protein_mask
                                         for current_cluster
                                         in current_clusters
                                         ], axis=None,
                                        )
                cluster_contact_mask = np.concatenate([current_cluster.cluster_contact_mask
                                         for current_cluster
                                         in current_clusters
                                         ], axis=None,
                                        )

                centroid_array = np.mean(np.array(cluster_positions_list), axis=0)

                centroid = (centroid_array[0],
                            centroid_array[1],
                            centroid_array[2],)

                # event_mask_indicies_list =

                # if len(event_mask_indicies_list) > 1:
                event_mask_indicies = tuple(
                    np.concatenate(
                        [current_cluster.event_mask_indicies[i]
                         for current_cluster
                         in current_clusters
                         if current_cluster.event_mask_indicies
                         ],
                        axis=None,
                    )
                    for i
                    in [0, 1, 2]
                )
                # else:
                # event_mask_indicies = None

                new_cluster = Cluster(
                    cluster_points_tuple,
                    cluster_positions_array,
                    values,
                    centroid,
                    event_mask_indicies,
                    cluster_inner_protein_mask,
                    cluster_contact_mask,
                )

                new_clusters[unique_cluster] = new_cluster

            new_clustering_dict[dtag] = Clustering(new_clusters)

        return Clusterings(new_clustering_dict)

    def filter_distance_from_protein(self):
        pass

    def group_close(self):
        pass

    def remove_symetry_pairs(self):
        pass

    def __getitem__(self, item):
        return self.clusterings[item]

    def __len__(self):
        return len(self.clusterings)

    def __iter__(self):
        for dtag in self.clusterings:
            yield dtag


@dataclasses.dataclass()
class BDC:
    bdc: float
    mean_fraction: float
    feature_fraction: float

    @staticmethod
    def from_float(bdc: float):
        pass

    @staticmethod
    def from_cluster(xmap: Xmap, model: Model, cluster: Cluster, dtag: Dtag, grid: Grid,
                     min_bdc=0.0, max_bdc=0.95, steps=100):
        xmap_array = xmap.to_array(copy=True)

        cluster_indexes = cluster.event_mask_indicies

        protein_mask = np.array(grid.partitioning.protein_mask, copy=False, dtype=np.int8)
        protein_mask_indicies = np.nonzero(protein_mask)

        xmap_masked = xmap_array[protein_mask_indicies]
        mean_masked = model.mean[protein_mask_indicies]
        cluster_array = np.full(protein_mask.shape, False)
        cluster_array[cluster_indexes] = True
        cluster_mask = cluster_array[protein_mask_indicies]

        vals = {}
        for val in np.linspace(min_bdc, max_bdc, steps):
            subtracted_map = xmap_masked - val * mean_masked
            cluster_vals = subtracted_map[cluster_mask]
            # local_correlation = stats.pearsonr(mean_masked[cluster_mask],
            #                                    cluster_vals)[0]
            # local_correlation, local_offset = np.polyfit(x=mean_masked[cluster_mask], y=cluster_vals, deg=1)
            local_correlation = np.corrcoef(x=mean_masked[cluster_mask], y=cluster_vals)[0, 1]

            # global_correlation = stats.pearsonr(mean_masked,
            #                                     subtracted_map)[0]
            # global_correlation, global_offset = np.polyfit(x=mean_masked, y=subtracted_map, deg=1)
            global_correlation = np.corrcoef(x=mean_masked, y=subtracted_map)[0, 1]

            vals[val] = np.abs(global_correlation - local_correlation)

        mean_fraction = max(vals,
                            key=lambda x: vals[x],
                            )

        return BDC(
            mean_fraction,
            mean_fraction,
            1 - mean_fraction,
        )


@dataclasses.dataclass()
class Event:
    event_id: EventID
    site: SiteID
    bdc: BDC
    cluster: Cluster
    native_centroid: Tuple[float, float, float]
    native_positions: List[Tuple[float, float, float]]

    @staticmethod
    def from_cluster(event_id: EventID,
                     cluster: Cluster,
                     site: SiteID,
                     bdc: BDC,
                     native_centroid: Tuple[float, float, float],
                     native_positions: List[Tuple[float, float, float]]):
        return Event(event_id=event_id,
                     site=site,
                     bdc=bdc,
                     cluster=cluster,
                     native_centroid=native_centroid,
                     native_positions=native_positions,
                     )


@dataclasses.dataclass()
class Events:
    events: typing.Dict[EventID, Event]
    sites: Sites

    @staticmethod
    def from_clusters(clusterings: Clusterings, model: Model, xmaps: Xmaps, grid: Grid,
                      alignment: Alignment, cutoff: float,
                      min_bdc, max_bdc,
                      mapper: Any = None):
        events: typing.Dict[EventID, Event] = {}

        sites: Sites = Sites.from_clusters(clusterings, cutoff)

        if mapper:
            jobs = {}
            for dtag in clusterings:
                clustering = clusterings[dtag]
                for event_idx in clustering:
                    event_idx = EventIDX(event_idx)
                    event_id = EventID(dtag, event_idx)

                    cluster = clustering[event_idx.event_idx]
                    xmap = xmaps[dtag]

                    site: SiteID = sites.event_to_site[event_id]

                    jobs[event_id] = delayed(Events.get_event)(xmap, cluster, dtag, site, event_id, model, grid,
                                                               min_bdc, max_bdc, )

            results = mapper(job for job in jobs.values())

            events = {event_id: event for event_id, event in zip(jobs.keys(), results)}

        else:
            for dtag in clusterings:
                clustering = clusterings[dtag]
                for event_idx in clustering:
                    event_idx = EventIDX(event_idx)
                    event_id = EventID(dtag, event_idx)

                    cluster = clustering[event_idx.event_idx]
                    xmap = xmaps[dtag]
                    bdc = BDC.from_cluster(xmap, model, cluster, dtag, grid, min_bdc, max_bdc, )

                    site: SiteID = sites.event_to_site[event_id]

                    # Get native centroid
                    native_centroid = alignment.reference_to_moving(
                        np.array(
                            (cluster.centroid[0],
                             cluster.centroid[1],
                             cluster.centroid[2],)).reshape(-1, 3)
                    )[0]

                    # Get native event mask
                    # event_positions = []
                    # for cluster_position in cluster.position_array:
                    #     position = grid.grid.point_to_position(grid.grid.get_point(x, y, z))
                    #     event_positions.append([position.x, position.y, position.z])

                    native_positions = alignment.reference_to_moving(cluster.cluster_positions_array)

                    event = Event.from_cluster(event_id,
                                               cluster,
                                               site,
                                               bdc,
                                               native_centroid,
                                               native_positions,
                                               )

                    events[event_id] = event

        return Events(events, sites)

    @staticmethod
    def get_event(xmap, cluster, dtag, site, event_id, model, grid, min_bdc, max_bdc, ):
        bdc = BDC.from_cluster(xmap, model, cluster, dtag, grid, min_bdc, max_bdc, )

        event = Event.from_cluster(
            event_id,
            cluster,
            site,
            bdc,
        )

        return event

    @staticmethod
    def from_all_events(event_dict: typing.Dict[EventID, Event], grid: Grid, cutoff: float):

        # Get the sites
        all_clusterings_dict = {}
        for event_id in event_dict:
            if event_id.dtag not in all_clusterings_dict:
                all_clusterings_dict[event_id.dtag] = {}

            all_clusterings_dict[event_id.dtag][event_id.event_idx.event_idx] = event_dict[event_id].cluster

        all_clusterings = {}
        for dtag in all_clusterings_dict:
            all_clusterings[dtag] = Clustering(all_clusterings_dict[dtag])

        clusterings = Clusterings(all_clusterings)

        sites: Sites = Sites.from_clusters(clusterings, cutoff)

        # Add sites to events
        events: typing.Dict[EventID, Event] = {}
        for event_id in event_dict:
            event = event_dict[event_id]

            for event_id_site, event_site in sites.event_to_site.items():
                if (event_id_site.dtag.dtag == event_id.dtag.dtag) and (
                        event_id_site.event_idx.event_idx == event_id.event_idx.event_idx):
                    site = event_site

            event.site = site

            events[event_id] = event

        return Events(events, sites)

    def __iter__(self):
        for event_id in self.events:
            yield event_id

    def __getitem__(self, item):
        return self.events[item]

    def save_event_maps(
            self,
            datasets,
            alignments,
            xmaps,
            model,
            pandda_fs_model,
            grid,
            structure_factors,
            outer_mask,
            inner_mask_symmetry,
            sample_rate,
            native_grid,
            mapper=False,
    ):

        processed_datasets = {}
        for event_id in self:
            dtag = event_id.dtag
            event = self[event_id]
            string = f"""
            dtag: {dtag}
            event bdc: {event.bdc}
            centroid: {event.cluster.centroid}
            """
            if dtag not in processed_datasets:
                processed_datasets[dtag] = pandda_fs_model.processed_datasets[event_id.dtag]

            processed_datasets[dtag].event_map_files.add_event(event)

        if mapper:
            event_id_list = list(self.events.keys())

            # Get unique dtags
            event_dtag_list = []
            for event_id in event_id_list:
                dtag = event_id.dtag

                if len(
                        list(
                            filter(
                                lambda event_dtag: event_dtag.dtag == dtag.dtag,
                                event_dtag_list,
                            )
                        )
                ) == 0:
                    event_dtag_list.append(dtag)

            results = mapper(
                delayed(
                    Partitioning.from_structure_multiprocess)(
                    datasets[dtag].structure,
                    # grid,
                    native_grid,
                    outer_mask,
                    inner_mask_symmetry,
                )
                for dtag
                in event_dtag_list
            )

            partitioning_dict = {dtag: partitioning for dtag, partitioning in zip(event_dtag_list, results)}

            results = mapper(
                delayed(
                    save_event_map)(
                    processed_datasets[event_id.dtag].event_map_files[event_id.event_idx].path,
                    xmaps[event_id.dtag],
                    model,
                    self[event_id],
                    datasets[event_id.dtag],
                    alignments[event_id.dtag],
                    grid,
                    structure_factors,
                    outer_mask,
                    inner_mask_symmetry,
                    partitioning_dict[event_id.dtag],
                    sample_rate,
                )
                for event_id
                in event_id_list
            )
