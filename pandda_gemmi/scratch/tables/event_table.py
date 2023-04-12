import dataclasses

import numpy as np
import pandas as pd

from ..interfaces import *


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
    def from_event(event):
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
    def from_events(events):
        records = []
        for event_id in events:
            event_record = EventTableRecord.from_event(events[event_id])
            records.append(event_record)

        return EventTable(records)

    def save(self, path: Path):
        records = []
        for record in self.records:
            event_dict = dataclasses.asdict(record)
            event_dict["1-BDC"] = round(1 - event_dict["bdc"], 2)
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
        # x=event.cluster.centroid[0],
        # y=event.cluster.centroid[1],
        # z=event.cluster.centroid[2],
        x=event.native_centroid[0],
        y=event.native_centroid[1],
        z=event.native_centroid[2],
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
        events,
        sites,
        event_ranking,
):
    records = []
    for event_id in event_ranking:
        event = events[event_id]
        site_id = sites.event_to_site[event_id]
        event_record = get_event_record_from_event_site(event, site_id)
        records.append(event_record)

    return EventTable(records)
