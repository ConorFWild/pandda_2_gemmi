import itertools

import numpy as np
from sklearn.cluster import DBSCAN
import gemmi

from .event import Event

from ..dmaps import SparseDMap


class ClusterDensityDBSCAN:
    # def __call__(self, z, reference_frame):
    #     z_grid = reference_frame.unmask(SparseDMap(z))
    #
    #     point_array = np.vstack(
    #         [partition.points for resid, partition in reference_frame.partitioning.partitions.items()]) % np.array(
    #         reference_frame.spacing)
    #     position_array = np.vstack(
    #         [partition.positions for resid, partition in reference_frame.partitioning.partitions.items()])
    #
    #     _all_points_array = point_array
    #     all_points_array = point_array
    #     all_positions_array = position_array
    #
    #     min_point = np.min(all_points_array, axis=0)
    #     max_point = np.max(all_points_array, axis=0)
    #
    #     all_point_indexes = (all_points_array[:, 0], all_points_array[:, 1], all_points_array[:, 2],)
    #     shape = reference_frame.spacing
    #     point_3d_array_x = np.zeros((shape[0], shape[1], shape[2]), )
    #     point_3d_array_y = np.zeros((shape[0], shape[1], shape[2]), )
    #     point_3d_array_z = np.zeros((shape[0], shape[1], shape[2]), )
    #
    #     point_3d_array_x[all_point_indexes] = all_points_array[:, 0]
    #     point_3d_array_y[all_point_indexes] = all_points_array[:, 1]
    #     point_3d_array_z[all_point_indexes] = all_points_array[:, 2]
    #
    #     pos_3d_arr_x = np.zeros((shape[0], shape[1], shape[2]))
    #     pos_3d_arr_y = np.zeros((shape[0], shape[1], shape[2]))
    #     pos_3d_arr_z = np.zeros((shape[0], shape[1], shape[2]))
    #
    #     pos_3d_arr_x[all_point_indexes] = all_positions_array[:, 0]
    #     pos_3d_arr_y[all_point_indexes] = all_positions_array[:, 1]
    #     pos_3d_arr_z[all_point_indexes] = all_positions_array[:, 2]
    #
    #     z_unmasked_array = np.array(z_grid, copy=False)
    #
    #     high_z_indexes = np.nonzero(z_unmasked_array > 2.0)
    #     high_z_pos_x = pos_3d_arr_x[high_z_indexes]
    #     high_z_pos_y = pos_3d_arr_y[high_z_indexes]
    #     high_z_pos_z = pos_3d_arr_z[high_z_indexes]
    #
    #     high_z_pos_array = np.hstack([
    #         high_z_pos_x.reshape((-1, 1)),
    #         high_z_pos_y.reshape((-1, 1)),
    #         high_z_pos_z.reshape((-1, 1))
    #     ])
    #
    #     if high_z_pos_array.shape[0] == 0:
    #         return {}
    #
    #     clusters = DBSCAN(eps=1.0, min_samples=5).fit_predict(high_z_pos_array)
    #
    #     high_z_point_x = point_3d_array_x[high_z_indexes]
    #     high_z_point_y = point_3d_array_y[high_z_indexes]
    #     high_z_point_z = point_3d_array_z[high_z_indexes]
    #
    #     high_z_point_array = np.hstack([
    #         high_z_point_x.reshape((-1, 1)),
    #         high_z_point_y.reshape((-1, 1)),
    #         high_z_point_z.reshape((-1, 1))
    #     ])
    #
    #     cluster_nums, counts = np.unique(clusters, return_counts=True)
    #
    #     events = {}
    #     j = 0
    #     for cluster_num in cluster_nums:
    #         if cluster_num == -1:
    #             continue
    #         events[j] = Event(
    #             high_z_pos_array[clusters == cluster_num, :],
    #             high_z_point_array[clusters == cluster_num, :].astype(np.int),
    #             0.0,
    #             0.0
    #         )
    #         j +=1
    #
    #     return events

    def __call__(self, z, reference_frame, debug=False):

        z_grid = reference_frame.unmask(SparseDMap(z))

        # Get the points and their respective positions associated with each CA
        # Points are in real space, i.e. not modulus
        point_array = np.vstack(
            [partition.points for resid, partition in reference_frame.partitioning.partitions.items()]) % np.array(
            reference_frame.spacing)
        position_array = np.vstack(
            [partition.positions for resid, partition in reference_frame.partitioning.partitions.items()])

        _all_points_array = point_array
        all_points_array = point_array
        all_positions_array = position_array

        # Get the minimum and maximum points
        min_point = np.min(all_points_array, axis=0).astype(int)
        max_point = np.max(all_points_array, axis=0).astype(int)

        # Get the points relative to the minimum point
        all_point_indexes = (
            all_points_array[:, 0] - min_point[0],
            all_points_array[:, 1] - min_point[1],
            all_points_array[:, 2] - min_point[2],)

        # Get the points modulused
        all_point_indexes_mod = (
            np.mod(all_points_array[:, 0], reference_frame.spacing[0]),
            np.mod(all_points_array[:, 1], reference_frame.spacing[1]),
            np.mod(all_points_array[:, 2], reference_frame.spacing[2]),
        )

        # Get the shape of the box contiaining the points
        shape = [(max_point[_j] - min_point[_j])+1 for _j in (0,1,2)]

        # Define three 3d arrays with the x, y and z of the point coordinates
        point_3d_array_x = np.zeros((shape[0], shape[1], shape[2]), )
        point_3d_array_y = np.zeros((shape[0], shape[1], shape[2]), )
        point_3d_array_z = np.zeros((shape[0], shape[1], shape[2]), )

        point_3d_array_x[all_point_indexes] = all_points_array[:, 0]
        point_3d_array_y[all_point_indexes] = all_points_array[:, 1]
        point_3d_array_z[all_point_indexes] = all_points_array[:, 2]

        # Same for positions
        pos_3d_arr_x = np.zeros((shape[0], shape[1], shape[2]))
        pos_3d_arr_y = np.zeros((shape[0], shape[1], shape[2]))
        pos_3d_arr_z = np.zeros((shape[0], shape[1], shape[2]))

        pos_3d_arr_x[all_point_indexes] = all_positions_array[:, 0]
        pos_3d_arr_y[all_point_indexes] = all_positions_array[:, 1]
        pos_3d_arr_z[all_point_indexes] = all_positions_array[:, 2]

        # Get the z grid as an array
        z_unmasked_array = np.array(z_grid, copy=False)


        # Get the high z values mask
        # high_z_indexes = np.nonzero(z_unmasked_array > 2.0)
        # high_z_all_points_mask = z_unmasked_array[all_point_indexes_mod] > 2.0
        # cutoff = max(np.quantile(z, 0.95), 2.0)
        cutoff = max(np.quantile(z, 0.98), 2.0)

        # print(f"Cutoff is: {cutoff}")
        high_z_all_points_mask = z_unmasked_array[all_point_indexes_mod] > cutoff


        # Get the indicies of high z values in real space i.e. not modulused
        # high_z_indexes = z_unmasked_array[all_point_indexes_mod] > 2.0
        high_z_indexes = (
            all_point_indexes[0][high_z_all_points_mask],
            all_point_indexes[1][high_z_all_points_mask],
            all_point_indexes[2][high_z_all_points_mask],
        )
        high_z_pos_x = pos_3d_arr_x[high_z_indexes]
        high_z_pos_y = pos_3d_arr_y[high_z_indexes]
        high_z_pos_z = pos_3d_arr_z[high_z_indexes]

        high_z_pos_array = np.hstack([
            high_z_pos_x.reshape((-1, 1)),
            high_z_pos_y.reshape((-1, 1)),
            high_z_pos_z.reshape((-1, 1))
        ])

        if high_z_pos_array.shape[0] == 0:
            return {}, {"cutoff": cutoff, "high_z_all_points_mask": high_z_all_points_mask, "eps": 0.0}

        initial_pos = gemmi.Position(0.0,0.0,0.0)
        dists = []
        for x, y, z in itertools.product([-1.0,1.0],
                                         [-1.0,1.0],
                                         [-1.0,1.0],
                                         ):
            frac = gemmi.Fractional(x/z_grid.nu, y/z_grid.nv, z/z_grid.nw)
            orth = z_grid.unit_cell.orthogonalize(frac)
            dist = orth.dist(initial_pos)
            dists.append(dist)

        eps = max(dists)*1.5
        # print(f"Got an eps of: {eps}")

        clusters = DBSCAN(
            eps=eps,
            min_samples=15,
        ).fit_predict(high_z_pos_array)

        high_z_point_x = point_3d_array_x[high_z_indexes]
        high_z_point_y = point_3d_array_y[high_z_indexes]
        high_z_point_z = point_3d_array_z[high_z_indexes]

        high_z_point_array = np.hstack([
            high_z_point_x.reshape((-1, 1)),
            high_z_point_y.reshape((-1, 1)),
            high_z_point_z.reshape((-1, 1))
        ])

        cluster_nums, counts = np.unique(clusters, return_counts=True)

        events = {}
        j = 0
        for cluster_num in cluster_nums:
            if cluster_num == -1:
                continue

            events[j] = Event(
                high_z_pos_array[clusters == cluster_num, :],
                high_z_point_array[clusters == cluster_num, :].astype(int),
                high_z_point_array[clusters == cluster_num, :].shape[0] * (z_grid.unit_cell.volume / z_grid.point_count),
                np.mean(high_z_pos_array[clusters == cluster_num, :], axis=0),
                0.0,
                0.0,
                local_strength=0.0
            )
            j +=1

        event_pos_list = []
        for event_idx, event in events.items():
            event_pos_list.append(event.centroid)

        events_array = np.array([event for event in events.values()])
        event_pos_array = np.array(event_pos_list)

        for event_num, event in events.items():
            event_dist_array = np.linalg.norm(
                event_pos_array-event.centroid.reshape(1,3),
                axis=1
            )
            # print(event_dist_array)
            close_events = events_array[event_dist_array < 6.0]
            # print(f"Close event: {close_events}")
            event.local_strength = np.sum([close_event.size for close_event in close_events])
            # print(f"Local strength: {}")


        return events, {"cutoff": cutoff, "high_z_all_points_mask": high_z_all_points_mask, "eps": eps}
