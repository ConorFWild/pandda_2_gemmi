from __future__ import annotations

import typing
import dataclasses

from scipy.cluster.hierarchy import fclusterdata
from joblib.externals.loky import set_loky_pickler

set_loky_pickler('pickle')

from pandda_gemmi.python_types import *
from pandda_gemmi.common import Dtag, EventID, EventIDX, SiteID, PositionsArray



@dataclasses.dataclass()
class Site:
    number: int
    centroid: typing.List[float]


@dataclasses.dataclass()
class Sites:
    site_to_event: typing.Dict[SiteID, typing.List[EventID]]
    event_to_site: typing.Dict[EventID, SiteID]
    centroids: typing.Dict[SiteID, np.ndarray]

    def __iter__(self):
        for site_id in self.site_to_event:
            yield site_id

    def __getitem__(self, item):
        return self.site_to_event[item]

    @staticmethod
    def from_clusters(clusterings,#: Clusterings,
                      cutoff: float):
        flat_clusters = {}
        for dtag in clusterings:
            for event_idx in clusterings[dtag]:
                event_idx = EventIDX(event_idx)
                flat_clusters[EventID(dtag, event_idx)] = clusterings[dtag][event_idx.event_idx]

        centroids: typing.List[gemmi.Position] = [cluster.centroid for cluster in flat_clusters.values()]
        positions_array = PositionsArray.from_positions(centroids)

        print(positions_array.to_array().shape)
        # If there is no events
        if positions_array.to_array().shape[0] == 0:
            return Sites({}, {}, {})

        # If there is one or less event
        if positions_array.to_array().shape[0] == 1:
            site_to_event = {}
            event_to_site = {}
            site_centroids = {}

            for site_id, event_id in enumerate(flat_clusters):
                site_id = SiteID(int(site_id))

                if not site_id in site_to_event:
                    site_to_event[site_id] = []

                site_to_event[site_id].append(event_id)
                event_to_site[event_id] = site_id

                cluster_coord_list = []
                cluster_centroid = flat_clusters[event_id].centroid
                cluster_coord_list.append(cluster_centroid)
                array = np.array(cluster_coord_list)
                mean_centroid = np.mean(array, axis=0)

                site_centroids[site_id] = mean_centroid

            return Sites(site_to_event, event_to_site, site_centroids)

        site_ids_array = fclusterdata(X=positions_array.to_array(),
                                      t=cutoff,
                                      criterion='distance',
                                      metric='euclidean',
                                      method='average',
                                      )

        site_to_event = {}
        event_to_site = {}
        for event_id, site_id in zip(flat_clusters, site_ids_array):
            site_id = SiteID(int(site_id))

            if not site_id in site_to_event:
                site_to_event[site_id] = []

            site_to_event[site_id].append(event_id)
            event_to_site[event_id] = site_id

        # Site centroids
        site_centroids = {}
        for site_id, event_id_list in site_to_event.items():
            cluster_coord_list = []
            for event_id in event_id_list:
                cluster_centroid = flat_clusters[event_id].centroid
                cluster_coord_list.append(cluster_centroid)
            array = np.array(cluster_coord_list)
            mean_centroid = np.mean(array, axis=0)
            site_centroids[site_id] = mean_centroid

        return Sites(site_to_event, event_to_site, site_centroids)
