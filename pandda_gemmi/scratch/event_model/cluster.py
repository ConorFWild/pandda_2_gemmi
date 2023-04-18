import numpy as np
from sklearn.cluster import DBSCAN

from .event import Event

from ..dmaps import SparseDMap


class ClusterDensityDBSCAN:
    def __call__(self, z, reference_frame):
        z_grid = reference_frame.unmask(SparseDMap(z))

        point_array = np.vstack(
            [partition.points for resid, partition in reference_frame.partitioning.partitions.items()]) % np.array(
            reference_frame.spacing)
        position_array = np.vstack(
            [partition.positions for resid, partition in reference_frame.partitioning.partitions.items()])

        _all_points_array = point_array
        all_points_array = point_array
        all_positions_array = position_array

        all_point_indexes = (all_points_array[:, 0], all_points_array[:, 1], all_points_array[:, 2],)
        shape = reference_frame.spacing
        point_3d_array_x = np.zeros((shape[0], shape[1], shape[2]), )
        point_3d_array_y = np.zeros((shape[0], shape[1], shape[2]), )
        point_3d_array_z = np.zeros((shape[0], shape[1], shape[2]), )

        point_3d_array_x[all_point_indexes] = all_points_array[:, 0]
        point_3d_array_y[all_point_indexes] = all_points_array[:, 1]
        point_3d_array_z[all_point_indexes] = all_points_array[:, 2]

        pos_3d_arr_x = np.zeros((shape[0], shape[1], shape[2]))
        pos_3d_arr_y = np.zeros((shape[0], shape[1], shape[2]))
        pos_3d_arr_z = np.zeros((shape[0], shape[1], shape[2]))

        pos_3d_arr_x[all_point_indexes] = all_positions_array[:, 0]
        pos_3d_arr_y[all_point_indexes] = all_positions_array[:, 1]
        pos_3d_arr_z[all_point_indexes] = all_positions_array[:, 2]

        z_unmasked_array = np.array(z_grid, copy=False)

        high_z_indexes = np.nonzero(z_unmasked_array > 2.0)
        high_z_pos_x = pos_3d_arr_x[high_z_indexes]
        high_z_pos_y = pos_3d_arr_y[high_z_indexes]
        high_z_pos_z = pos_3d_arr_z[high_z_indexes]

        high_z_pos_array = np.hstack([
            high_z_pos_x.reshape((-1, 1)),
            high_z_pos_y.reshape((-1, 1)),
            high_z_pos_z.reshape((-1, 1))
        ])

        if high_z_pos_array.shape[0] == 0:
            return {}

        clusters = DBSCAN(eps=1.0, min_samples=5).fit_predict(high_z_pos_array)

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
                high_z_point_array[clusters == cluster_num, :].astype(np.int),
                0.0,
                0.0
            )
            j +=1

        return events
