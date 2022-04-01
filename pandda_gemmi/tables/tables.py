from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import *

import numpy as np
from joblib.externals.loky import set_loky_pickler

from pandda_gemmi.analyse_interface import EventsInterface, GetEventTableInterface, SitesInterface

set_loky_pickler('pickle')

import pandas as pd

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.common import SiteID
from pandda_gemmi.sites import Sites
from pandda_gemmi.event import Event, Events, Clustering, Clusterings


@dataclasses.dataclass()
class EventTableRecord:
    dtag: str
    event_idx: int
    bdc: float
    cluster_size: int
    global_correlation_to_average_map: float
    global_correlation_to_mean_map: float
    local_correlation_to_average_map: float
    local_correlation_to_mean_map: float
    site_idx: int
    x: float
    y: float
    z: float
    z_mean: float
    z_peak: float
    applied_b_factor_scaling: float
    high_resolution: float
    low_resolution: float
    r_free: float
    r_work: float
    analysed_resolution: float
    map_uncertainty: float
    analysed: bool
    interesting: bool
    exclude_from_z_map_analysis: bool
    exclude_from_characterisation: bool

    @staticmethod
    def from_event(event: Event):
        return EventTableRecord(
            dtag=event.event_id.dtag.dtag,
            event_idx=event.event_id.event_idx.event_idx,
            bdc=event.bdc.bdc,
            cluster_size=event.cluster.values.size,
            global_correlation_to_average_map=0,
            global_correlation_to_mean_map=0,
            local_correlation_to_average_map=0,
            local_correlation_to_mean_map=0,
            site_idx=event.site.site_id,
            x=event.cluster.centroid[0],
            y=event.cluster.centroid[1],
            z=event.cluster.centroid[2],
            z_mean=0.0,
            z_peak=0.0,
            applied_b_factor_scaling=0.0,
            high_resolution=0.0,
            low_resolution=0.0,
            r_free=0.0,
            r_work=0.0,
            analysed_resolution=0.0,
            map_uncertainty=0.0,
            analysed=False,
            interesting=False,
            exclude_from_z_map_analysis=False,
            exclude_from_characterisation=False,
        )


@dataclasses.dataclass()
class EventTable:
    records: List[EventTableRecord]

    @staticmethod
    def from_events(events: Events):
        records = []
        for event_id in events:
            event_record = EventTableRecord.from_event(events[event_id])
            records.append(event_record)

        return EventTable(records)

    def save(self, path: Path):
        records = []
        for record in self.records:
            event_dict = dataclasses.asdict(record)
            event_dict["1-BDC"] = round(1-event_dict["bdc"], 2)
            records.append(event_dict)
        table = pd.DataFrame(records)
        table.to_csv(str(path))


def get_event_record_from_event_site(event: EventInterface, site_id: SiteIDInterface) -> EventTableRecord:
    return EventTableRecord(
                dtag=event.event_id.dtag.dtag,
                event_idx=event.event_id.event_idx.event_idx,
                bdc=event.bdc.bdc,
                cluster_size=event.cluster.values.size,
                global_correlation_to_average_map=0,
                global_correlation_to_mean_map=0,
                local_correlation_to_average_map=0,
                local_correlation_to_mean_map=0,
                site_idx=site_id.site_id,
                x=event.cluster.centroid[0],
                y=event.cluster.centroid[1],
                z=event.cluster.centroid[2],
                z_mean=0.0,
                z_peak=0.0,
                applied_b_factor_scaling=0.0,
                high_resolution=0.0,
                low_resolution=0.0,
                r_free=0.0,
                r_work=0.0,
                analysed_resolution=0.0,
                map_uncertainty=0.0,
                analysed=False,
                interesting=False,
                exclude_from_z_map_analysis=False,
                exclude_from_characterisation=False,
            )

def get_event_table_from_events(
    events: EventsInterface, 
    sites: SitesInterface, 
    event_ranking: EventRankingInterface,
    ) -> EventTableInterface:
    records = []
    for event_id in event_ranking:
        event = events[event_id]
        site_id = sites.event_to_site[event_id]
        event_record = get_event_record_from_event_site(event, site_id)
        records.append(event_record)

    return EventTable(records)


class GetEventTable(GetEventTableInterface):
    def __call__(self, events: EventsInterface, sites: SitesInterface, event_ranking: EventRankingInterface) -> EventTableInterface:
        return get_event_table_from_events(events, sites, event_ranking)


@dataclasses.dataclass()
class SiteTableRecord:
    site_idx: int
    centroid: Tuple[float, float, float]

    @staticmethod
    def from_site_id(site_id: SiteID, centroid: np.ndarray):
        return SiteTableRecord(
            site_idx=site_id.site_id,
            centroid=(centroid[0], centroid[1], centroid[2],),
        )


@dataclasses.dataclass()
class SiteTable:
    site_record_list: List[SiteTableRecord]

    def __iter__(self):
        for record in self.site_record_list:
            yield record

    @staticmethod
    def from_events(events: Events, cutoff: float):

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

def get_site_table_from_events(events: EventsInterface, initial_sites: SitesInterface, cutoff: float):

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

class GetSiteTable(GetSiteTableInterface):
    def __call__(self, events: EventsInterface, sites: SitesInterface, cutoff: float) -> SiteTableInterface:
        return get_site_table_from_events(events, sites, cutoff)
