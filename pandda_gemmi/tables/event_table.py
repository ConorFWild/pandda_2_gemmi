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
    hit_in_site_probability: float
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
    def from_event(dataset: DatasetInterface, event_id, event: EventInterface, site_id, hit_in_site_probability):
        # centroid = np.mean(event.pos_array, axis=0)
        centroid = event.centroid
        return EventTableRecord(
            dtag=event_id[0],
            event_idx=event_id[1],
            bdc=event.bdc,
            cluster_size=event.pos_array.shape[0],
            global_correlation_to_average_map=0,
            global_correlation_to_mean_map=0,
            local_correlation_to_average_map=0,
            local_correlation_to_mean_map=0,
            site_idx=site_id,
            x=centroid[0],
            y=centroid[1],
            z=centroid[2],
            # z_mean=0.0,
            z_mean=float(event.score),
            z_peak=float(event.build.score),
            hit_in_site_probability=hit_in_site_probability,
            applied_b_factor_scaling=0.0,
            high_resolution=round(dataset.reflections.resolution(), 2),
            low_resolution=0.0,
            r_free=round(dataset.structure.rfree(), 2),
            r_work=round(dataset.structure.rwork(), 2),
            analysed_resolution=round(dataset.reflections.resolution(), 2),
            map_uncertainty=float('nan'),
            analysed=False,
            interesting=False,
            exclude_from_z_map_analysis=False,
            exclude_from_characterisation=False,
        )


@dataclasses.dataclass()
class EventTable:
    records: List[EventTableRecord]

    @staticmethod
    def from_events(datasets: Dict[str, DatasetInterface], events: Dict[Tuple[str, int], EventInterface], ranking, sites, hit_in_site_probabilities):
        records = []
        for event_id in ranking:
            for _site_id, site in sites.items():
                if event_id in site.event_ids:
                    site_id = _site_id
            event_record = EventTableRecord.from_event(datasets[event_id[0]], event_id, events[event_id], site_id, hit_in_site_probabilities[event_id])
            records.append(event_record)

        return EventTable(records)

    def save(self, path: Path):
        records = []
        for record in self.records:
            event_dict = dataclasses.asdict(record)
            event_dict["1-BDC"] = round(1 - event_dict["bdc"], 2)
            event_dict["x"] = round(event_dict['x'], 3)
            event_dict["y"] = round(event_dict['y'], 3)
            event_dict["z"] = round(event_dict['z'], 3)
            records.append(event_dict)
        table = pd.DataFrame(records)
        table.to_csv(str(path))


# def get_event_record_from_event_site(event: EventInterface, site_id) -> EventTableRecord:
#     return EventTableRecord(
#         dtag=event.event_id.dtag.dtag,
#         event_idx=event.event_id.event_idx.event_idx,
#         bdc=event.bdc.bdc,
#         cluster_size=event.cluster.values.size,
#         global_correlation_to_average_map=0,
#         global_correlation_to_mean_map=0,
#         local_correlation_to_average_map=0,
#         local_correlation_to_mean_map=0,
#         site_idx=site_id.site_id,
#         # x=event.cluster.centroid[0],
#         # y=event.cluster.centroid[1],
#         # z=event.cluster.centroid[2],
#         x=event.native_centroid[0],
#         y=event.native_centroid[1],
#         z=event.native_centroid[2],
#         z_mean=0.0,
#         z_peak=0.0,
#         applied_b_factor_scaling=0.0,
#         high_resolution=0.0,
#         low_resolution=0.0,
#         r_free=0.0,
#         r_work=0.0,
#         analysed_resolution=0.0,
#         map_uncertainty=0.0,
#         analysed=False,
#         interesting=False,
#         exclude_from_z_map_analysis=False,
#         exclude_from_characterisation=False,
#     )
#
#
# def get_event_table_from_events(
#         events,
#         sites,
#         event_ranking,
# ):
#     records = []
#     for event_id in event_ranking:
#         event = events[event_id]
#         site_id = sites.event_to_site[event_id]
#         event_record = get_event_record_from_event_site(event_id, event, site_id)
#         records.append(event_record)
#
#     return EventTable(records)

@dataclasses.dataclass()
class InspectEventTableRecord:
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
    hit_in_site_probability: float
    applied_b_factor_scaling: float
    high_resolution: float
    low_resolution: float
    r_free: float
    r_work: float
    analysed_resolution: float
    map_uncertainty: float
    analysed: bool
    exclude_from_z_map_analysis: bool
    exclude_from_characterisation: bool
    interesting: bool
    ligand_placed: bool
    ligand_confidence: str
    comment: str
    viewed: bool


    @staticmethod
    def from_event(dataset: DatasetInterface, event_id, event: EventInterface, site_id, hit_in_site_probability):
        # centroid = np.mean(event.pos_array, axis=0)
        centroid = event.centroid
        return InspectEventTableRecord(
            dtag=event_id[0],
            event_idx=event_id[1],
            bdc=event.bdc,
            cluster_size=event.pos_array.shape[0],
            global_correlation_to_average_map=0,
            global_correlation_to_mean_map=0,
            local_correlation_to_average_map=0,
            local_correlation_to_mean_map=0,
            site_idx=site_id,
            x=centroid[0],
            y=centroid[1],
            z=centroid[2],
            # z_mean=0.0,
            z_mean=float(event.score),
            z_peak=float(event.build.score),
            hit_in_site_probability=hit_in_site_probability,
            applied_b_factor_scaling=0.0,
            high_resolution=round(dataset.reflections.resolution(), 2),
            low_resolution=0.0,
            r_free=round(dataset.structure.rfree(), 2),
            r_work=round(dataset.structure.rwork(), 2),
            analysed_resolution=round(dataset.reflections.resolution(), 2),
            map_uncertainty=float('nan'),
            analysed=False,
            exclude_from_z_map_analysis=False,
            exclude_from_characterisation=False,
            interesting=event.interesting,
            ligand_placed=event.ligand_placed,
            ligand_confidence=event.ligand_confidence,
            comment=event.comment,
            viewed=event.viewed,
            )


@dataclasses.dataclass()
class InspectEventTable:
    records: List[InspectEventTableRecord]

    @staticmethod
    def from_events(datasets: Dict[str, DatasetInterface], events: Dict[Tuple[str, int], EventInterface], ranking, sites, hit_in_site_probabilities):
        records = []
        for event_id in ranking:
            for _site_id, site in sites.items():
                if event_id in site.event_ids:
                    site_id = _site_id
            event_record = InspectEventTableRecord.from_event(datasets[event_id[0]], event_id, events[event_id], site_id, hit_in_site_probabilities[event_id])
            records.append(event_record)

        return InspectEventTable(records)

    def save(self, path: Path):
        records = []
        for record in self.records:
            event_dict = dataclasses.asdict(record)
            event_dict["1-BDC"] = round(1 - event_dict["bdc"], 2)
            event_dict["x"] = round(event_dict['x'], 3)
            event_dict["y"] = round(event_dict['y'], 3)
            event_dict["z"] = round(event_dict['z'], 3)
            event_dict["Interesting"] = event_dict['interesting']
            event_dict["Ligand Placed"] = event_dict['ligand_placed']
            event_dict["Ligand Confidence"]= event_dict['ligand_confidence']
            event_dict["Comment"]= event_dict['comment']
            event_dict["Viewed"]= event_dict['viewed']

            records.append(event_dict)
        table = pd.DataFrame(records)
        table.to_csv(str(path))

