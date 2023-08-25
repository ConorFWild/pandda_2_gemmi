import numpy as np
from scipy.cluster.hierarchy import fclusterdata
import gemmi

from ..interfaces import *
from pandda_gemmi.processor import Partial
from pandda_gemmi.alignment import Alignment

from .site import Site


# class HeirarchicalSiteModel:
#
#     def __init__(self, t=8.0):
#         self.t = t
#
#     def __call__(self, events: Dict[Tuple[str, int], EventInterface]):
#
#         #
#         if len(events) == 0:
#             return {}
#
#         if len(events) == 1:
#             return {0: Site(
#                 list(events.keys())[0],
#                 np.mean(list(events.values())[0].pos_array, axis=1)
#             )}
#
#         # Get the array of centroids
#         centroid_array = np.array(
#             [
#                 np.mean(event.pos_array, axis=0)
#                 for event
#                 in events.values()
#             ]
#         )
#
#         #
#         clusters = fclusterdata(
#             centroid_array,
#             t=self.t,
#             criterion="distance",
#             method="centroid"
#         )
#
#         #
#         unique_clusters = np.unique(clusters, )
#
#         # Merge the events based on the clustering
#         event_array = np.array([event_id for event_id in events.keys()])
#         j = 0
#         sites = {}
#         for cluster_num in unique_clusters:
#             site_event_ids = event_array[clusters == cluster_num]
#             # print([event_id for event_id in site_event_ids])
#             site_positions = np.concatenate([events[(str(event_id[0]), int(event_id[1]))].pos_array for event_id in site_event_ids], axis=0)
#
#             sites[j] = Site(
#                 [(str(event_id[0]), int(event_id[1])) for event_id in site_event_ids],
#                 np.mean(site_positions, axis=0),
#             )
#             j += 1
#
#         return sites
def _get_closest_transform(unaligned_centroid, transforms, com_ref, com_mov):
    distances = []
    for transform, com_ref_centroid, com_mov_centroid in zip(transforms, com_ref, com_mov):
        distances.append(
            gemmi.Position(*com_mov_centroid).dist(gemmi.Position(*unaligned_centroid))
        )

    min_dist_index = np.argmin(distances)
    return transforms[min_dist_index], com_ref[min_dist_index, :], com_mov[min_dist_index, :]


class HeirarchicalSiteModel:

    def __init__(self, t=8.0):
        self.t = t

    def __call__(
            self,
            datasets: Dict[str, DatasetInterface],
            events: Dict[Tuple[str, int], EventInterface],
            processor: ProcessorInterface,
            structure_array_refs,
    ):

        #
        if len(events) == 0:
            return {}

        if len(events) == 1:
            return {0: Site(
                list(events.keys())[0],
                np.mean(list(events.values())[0].pos_array, axis=1)
            )}

        # Choose a reference structure from highest res dataset
        reference_dtag = min(datasets, key=lambda _dtag: datasets[_dtag].reflections.resolution())

        # Get alignments to this reference
        alignments: Dict[str, AlignmentInterface] = processor.process_dict(
            {_dtag: Partial(Alignment.from_structure_arrays).paramaterise(
                _dtag,
                structure_array_refs[_dtag],
                structure_array_refs[reference_dtag],
            ) for _dtag in datasets}
        )

        # Get the aligned centroids
        aligned_centroids = []
        for event_id, event in events.items():
            _dtag, _event_idx = event_id
            unaligned_centroid = np.mean(event.pos_array, axis=0)
            alignment = alignments[_dtag]
            transforms, com_ref, com_mov = alignment.get_transforms()
            transform, associated_com_ref, closest_com_mov = _get_closest_transform(
                unaligned_centroid,
                transforms,
                com_ref,
                com_mov,
            )
            transformed_pos = transform.apply(gemmi.Position(*(unaligned_centroid - closest_com_mov)))
            transformed_pos_np = np.array(
                [transformed_pos.x, transformed_pos.y, transformed_pos.z]) + associated_com_ref
            aligned_centroids.append(transformed_pos_np.flatten())

        # Get the array of centroids
        centroid_array = np.array(
            aligned_centroids
        )

        #
        clusters = fclusterdata(
            centroid_array,
            t=self.t,
            criterion="distance",
            method="centroid"
        )

        #
        unique_clusters = np.unique(clusters, )

        # Merge the events based on the clustering
        event_array = np.array([event_id for event_id in events.keys()])
        j = 0
        sites = {}
        for cluster_num in unique_clusters:
            site_event_ids = event_array[clusters == cluster_num]
            # print([event_id for event_id in site_event_ids])
            site_positions = np.concatenate(
                [events[(str(event_id[0]), int(event_id[1]))].pos_array for event_id in site_event_ids], axis=0)

            sites[j] = Site(
                [(str(event_id[0]), int(event_id[1])) for event_id in site_event_ids],
                np.mean(site_positions, axis=0),
            )
            j += 1

        return sites
