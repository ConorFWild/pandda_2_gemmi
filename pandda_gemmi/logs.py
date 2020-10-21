from __future__ import annotations
import json

from typing import Dict, List
import dataclasses

from pprint import PrettyPrinter
printer = PrettyPrinter(indent=1)
from pathlib import Path

from pandda_gemmi.pandda_types import *
from pandda_gemmi.python_types import *
from pandda_gemmi.config import Config

# @dataclasses.dataclass()
# class Log:
#     out_file: Path
#     log_dict: Dict

#     @staticmethod
#     def from_dir(out_dir: Path) -> Log:
#         pass
    
#     def log(self, message_dict: Dict):
#         self.log_dict.update(message_dict)

# class XmapLogs:
#     xmap_logs: Dict[str, str]

#     @staticmethod
#     def from_xmaps(xmaps: Xmaps):
#         logs = {}
#         for dtag in xmaps:
#             logs[dtag] = {}


# class ModelLogs:

#     @staticmethod
#     def from_model(model: Model):
#         logs_str = """
#         Model Summary
#             Per model stds
#                 {model_stds}
#         """
#         logs_str.format(model_stds=printer.pformat(model.stds))

#         return logs_str
    


@dataclasses.dataclass()
class DatasetLog:
    structure_path: str
    reflections_path: str
    
    @staticmethod
    def from_dataset(dataset: Dataset):
        return DatasetLog(
                structure_path=str(dataset.structure.path),
                reflections_path=str(dataset.reflections.path),
        )


@dataclasses.dataclass()
class InitialDatasetLog:
    datasets: Dict[str, DatasetLog]
    
    @staticmethod
    def from_initial_datasets(initial_datasets: Datasets):
        return {dtag.dtag: DatasetLog.from_dataset(initial_datasets[dtag])
                for dtag
                in initial_datasets
                }

@dataclasses.dataclass()
class InvalidDatasetLog:
    rejected_datasets: List[str]
    
    @staticmethod
    def from_datasets(datasets: Datasets, filtered_datasets: Datasets):
        rejected_dataset_dtag_list = []
        for dtag in datasets:
            if dtag in filtered_datasets.datasets:
                continue
            else:
                rejected_dataset_dtag_list.append(dtag.dtag)
    
        return rejected_dataset_dtag_list
    
@dataclasses.dataclass()
class LowResDatasetLog:
    rejected_datasets: List[str]

    @staticmethod
    def from_datasets(datasets: Datasets, filtered_datasets: Datasets):
        rejected_dataset_dtag_list = []
        for dtag in datasets:
            if dtag in filtered_datasets.datasets:
                continue
            else:
                rejected_dataset_dtag_list.append(dtag.dtag)
    
        return rejected_dataset_dtag_list
    
@dataclasses.dataclass()
class RFreeDatasetLog:
    rejected_datasets: List[str]
    
    @staticmethod
    def from_datasets(datasets: Datasets, filtered_datasets: Datasets):
        rejected_dataset_dtag_list = []
        for dtag in datasets:
            if dtag in filtered_datasets.datasets:
                continue
            else:
                rejected_dataset_dtag_list.append(dtag.dtag)
    
        return rejected_dataset_dtag_list

@dataclasses.dataclass()
class WilsonDatasetLog:
    rejected_datasets: List[str]
    
    @staticmethod
    def from_datasets(datasets: Datasets, filtered_datasets: Datasets):
        rejected_dataset_dtag_list = []
        for dtag in datasets:
            if dtag in filtered_datasets.datasets:
                continue
            else:
                rejected_dataset_dtag_list.append(dtag.dtag)
    
        return rejected_dataset_dtag_list

@dataclasses.dataclass()
class SmoothingDatasetLog:
    smoothing_factors: Dict[str, float]
    
    @staticmethod
    def from_datasets(datasets: Datasets):
        return {
            dtag.dtag: datasets[dtag].smoothing_factor
                for dtag
                in datasets
                }

@dataclasses.dataclass()
class StrucDatasetLog:
    rejected_datasets: List[str]
    
    @staticmethod
    def from_datasets(datasets: Datasets, filtered_datasets: Datasets):
        rejected_dataset_dtag_list = []
        for dtag in datasets:
            if dtag in filtered_datasets.datasets:
                continue
            else:
                rejected_dataset_dtag_list.append(dtag.dtag)
    
        return rejected_dataset_dtag_list

@dataclasses.dataclass()
class SpaceDatasetLog:
    rejected_datasets: List[str]

    @staticmethod
    def from_datasets(datasets: Datasets, filtered_datasets: Datasets):
        rejected_dataset_dtag_list = []
        for dtag in datasets:
            if dtag in filtered_datasets.datasets:
                continue
            else:
                rejected_dataset_dtag_list.append(dtag.dtag)
    
        return rejected_dataset_dtag_list


@dataclasses.dataclass()
class ClusterLog:
    centroid: List[float]
    size: float
    
    @staticmethod
    def from_cluster(cluster: Cluster, grid: Grid):
        return ClusterLog(
            centroid= cluster.centroid,
            size = cluster.size(grid),
        )

@dataclasses.dataclass()
class ClusteringsLog:
    
    @staticmethod
    def from_clusters(clusterings: Clusterings, grid: Grid):
        
        clusters = {}
        for dtag in clusterings:
            clustering = clusterings[dtag]
            for cluster_id in clustering:
                dtag_python = dtag.dtag
                cluster_id_python = cluster_id
                cluster = clusterings[dtag][cluster_id]
                if dtag_python not in clusters:
                    clusters[dtag_python] = {}
                
                clusters[dtag_python][cluster_id_python] = ClusterLog.from_cluster(cluster, 
                                                                                   grid,
                                                                                   )

        return clusters
    
@dataclasses.dataclass()
class EventLog:
    dtag: str
    idx: int
    centroid: List[float]
    bdc: float
    size: int
    
    @staticmethod
    def from_event(event: Event, grid:Grid):
        return EventLog(
            dtag = event.event_id.dtag.dtag,
            idx = event.event_id.event_idx.event_idx,
            centroid=event.cluster.centroid,
            bdc=event.bdc.bdc,
            size=event.cluster.size(grid),
        )

@dataclasses.dataclass()
class EventsLog:
    dataset_events: Dict[int, EventLog]
    
    @staticmethod
    def from_events(events: Events, grid: Grid):
        
        dtag_events = {}
        for event_id in events:
            dtag = event_id.dtag.dtag
            event_idx = event_id.event_idx.event_idx
            event = events[event_id]
            
            if dtag not in dtag_events:
                dtag_events[dtag] = {}
                
            dtag_events[dtag][event_idx] = EventLog.from_event(event, grid)
        
        return dtag_events

@dataclasses.dataclass()
class FSLog:
    out_dir: str
    
    @staticmethod
    def from_pandda_fs_model(pandda_fs: PanDDAFSModel):
        return FSLog(out_dir=str(pandda_fs.pandda_dir),
                     )
    
@dataclasses.dataclass()
class ReferenceLog:
    dtag: str
    
    @staticmethod
    def from_reference(reference: Reference):
        return ReferenceLog(reference.dtag.dtag)

@dataclasses.dataclass()
class GridLog:
    grid_size: List[float]
    unit_cell: UnitCellPython
    spacegroup: SpacegroupPython
    
    @staticmethod
    def from_grid(grid: Grid):
        unit_cell = UnitCellPython.from_gemmi(grid.grid.unit_cell)
        space_group = SpacegroupPython.from_gemmi(grid.grid.spacegroup)
        
        return GridLog(grid.shape(),
                       unit_cell,
                       space_group,
                       )
    
@dataclasses.dataclass()
class AlignmentLog:
    @staticmethod
    def from_alignment(alignment: Alignment):
        return None
    
@dataclasses.dataclass()
class AlignmentsLog:
    @staticmethod
    def from_alignments(alignments: Alignments):
        return {dtag.dtag: AlignmentLog.from_alignment(alignments[dtag])
                for dtag
                in alignments
                }
    
    
@dataclasses.dataclass()
class SiteLog:
    idx: int
    centroid: Tuple[float, float, float]
    
    @staticmethod
    def from_site_table_record(site_table_record: SiteTableRecord):
        return SiteLog(site_table_record.site_idx,
                       site_table_record.centroid,
                       )

@dataclasses.dataclass()
class SitesLog:
    @staticmethod
    def from_sites(site_table: SiteTable):
        return {site_table_record.site_idx: SiteLog.from_site_table_record(site_table_record)
                for site_table_record
                in site_table
                }


@dataclasses.dataclass()
class PreprocessingLog:
    initial_datasets_log: Dict[str, DatasetLog]
    invalid_datasets_log: List[str]
    low_res_datasets_log: List[str]
    rfree_datasets_log: List[str]
    wilson_datasets_log: List[str]
    smoothing_datasets_log: Dict[str, float]
    struc_datasets_log: List[str]
    space_datasets_log: List[str]
    
    @staticmethod
    def initialise():
        return PreprocessingLog(
            initial_datasets_log={},
            invalid_datasets_log=[],
            low_res_datasets_log=[],
            rfree_datasets_log=[],
            wilson_datasets_log=[],
            smoothing_datasets_log={},
            struc_datasets_log=[],
            space_datasets_log=[],
        )


@dataclasses.dataclass()
class ShellLog:
    number: int
    resolution: float
    test_dtags: List[str]
    train_dtags: List[str]
    sigma_is: Dict[str, float]
    initial_clusters: Dict[str, Dict[int, ClusterLog]]
    large_clusters: Dict[str, Dict[int, ClusterLog]]
    peaked_clusters: Dict[str, Dict[int, ClusterLog]]
    events: Dict[str, Dict[int, EventLog]]
    
    @staticmethod
    def from_shell(shell: Shell):
        return ShellLog(
            number=shell.number,
            resolution=shell.res_min.to_float(),
            test_dtags=[dtag.dtag for dtag in shell.test_dtags],
            train_dtags=[dtag.dtag for dtag in shell.train_dtags],
            sigma_is={},
            initial_clusters={},
            large_clusters={},
            peaked_clusters={},
            events={},
        )
    

@dataclasses.dataclass()
class LogData:
    config: Config
    fs_log: FSLog
    preprocessing_log: PreprocessingLog
    reference_log: ReferenceLog
    grid_log: GridLog
    alignments_log: Dict[str, AlignmentLog]
    shells_log: Dict[int, ShellLog]
    events_log: Dict[str, Dict[int, EventLog]]
    sites_log: Dict[int, SiteLog]    

    @staticmethod
    def initialise():
        preprocessing_log = PreprocessingLog.initialise()
        
        return LogData(
            config=None,
            fs_log=None,
            preprocessing_log=preprocessing_log,
            reference_log=None,
            grid_log=None,
            alignments_log={},
            shells_log={},
            events_log={},
            sites_log={},
                       )
        
    def save(self, path: Path):
        log_dict = dataclasses.asdict(self)
        
        with open(str(path), "w") as f:
            json.dump(log_dict,
                    f,
                    )
            
        