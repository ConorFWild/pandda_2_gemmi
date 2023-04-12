import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from ..interfaces import *

from .site import Site


class HeirarchicalSiteModel:

    def __init__(self, t=8.0):
        self.t = t

    def __call__(self, events: Dict[Tuple[str, int], EventInterface]):

        #
        if len(events) == 0:
            return {}

        if len(events) == 1:
            return {0: Site(
                list(events.keys())[0],
                np.mean(list(events.values())[0].pos_array, axis=1)
            )}

        # Get the array of centroids
        centroid_array = np.array(
            [
                np.mean(event.pos_array, axis=0)
                for event
                in events.values()
            ]
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
        event_array = np.array([event for event in events.values()])
        j = 0
        sites = {}
        for cluster_num in unique_clusters:
            site_events = event_array[clusters == cluster_num]
            site_positions = np.concatenate([event.pos_array for event in site_events])

            sites[j] = Site(
                [event_id for event_id in site_events],
                np.mean(site_positions, axis=0),
            )
            j += 1

        return sites

    ...
