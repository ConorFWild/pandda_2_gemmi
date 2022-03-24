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
