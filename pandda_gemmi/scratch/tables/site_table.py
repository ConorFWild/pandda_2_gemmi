import dataclasses

import numpy as np
import pandas as pd

from ..interfaces import *


@dataclasses.dataclass()
class SiteTableRecord:
    site_idx: int
    centroid: Tuple[float, float, float]

    @staticmethod
    def from_site_id(site_id, centroid: np.ndarray):
        return SiteTableRecord(
            site_idx=site_id,
            centroid=(centroid[0], centroid[1], centroid[2],),
        )


@dataclasses.dataclass()
class SiteTable:
    site_record_list: List[SiteTableRecord]

    def __iter__(self):
        for record in self.site_record_list:
            yield record

    @classmethod
    def from_sites(cls, sites):
        records = []
        for site_id, site in sites.items():
            # site = sites[site_id]
            centroid = site.centroid
            site_record = SiteTableRecord.from_site_id(site_id, centroid)
            records.append(site_record)

        return cls(records)

    @staticmethod
    def from_events(events, cutoff: float):

        dtag_clusters = {}
        for event_id in events:
            dtag = event_id.dtag
            event_idx = event_id.event_idx.event_idx
            event = events[event_id]

            if dtag not in dtag_clusters:
                dtag_clusters[dtag] = {}

            dtag_clusters[dtag][event_idx] = event.cluster

        _clusterings = {}
        for dtag in dtag_clusters:
            _clusterings[dtag] = Clustering(dtag_clusters[dtag])

        clusterings = Clusterings(_clusterings)

        sites: Sites = Sites.from_clusters(clusterings, cutoff)

        records = []
        for site_id in sites:
            # site = sites[site_id]
            centroid = sites.centroids[site_id]
            site_record = SiteTableRecord.from_site_id(site_id, centroid)
            records.append(site_record)

        return SiteTable(records)

    def save(self, path: Path):
        records = []
        for site_record in self.site_record_list:
            site_record_dict = dataclasses.asdict(site_record)
            records.append(site_record_dict)

        table = pd.DataFrame(records)

        table.to_csv(str(path))


def get_site_table_from_events(events, initial_sites, cutoff: float):
    dtag_clusters = {}
    for event_id in events:
        dtag = event_id.dtag
        event_idx = event_id.event_idx.event_idx
        event = events[event_id]

        if dtag not in dtag_clusters:
            dtag_clusters[dtag] = {}

        dtag_clusters[dtag][event_idx] = event.cluster

    _clusterings = {}
    for dtag in dtag_clusters:
        _clusterings[dtag] = Clustering(dtag_clusters[dtag])

    clusterings = Clusterings(_clusterings)

    sites: SitesInterface = Sites.from_clusters(clusterings, cutoff)

    records = []
    for site_id in sites:
        # site = sites[site_id]
        centroid = sites.centroids[site_id]
        site_record = SiteTableRecord.from_site_id(site_id, centroid)
        records.append(site_record)

    return SiteTable(records)