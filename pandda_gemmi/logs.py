from __future__ import annotations
import json

from typing import Dict, List
import dataclasses

from pprint import PrettyPrinter

printer = PrettyPrinter(indent=1)
from pathlib import Path, PosixPath

import numpy as np
import gemmi

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


def remove_paths(d):
    for k, v in d.items():
        if isinstance(v, dict):
            remove_paths(v)
        elif isinstance(v, PosixPath):
            d[k] = str(v)


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
            dtag.dtag: float(datasets[dtag].smoothing_factor)
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
            centroid=[float(x) for x in cluster.centroid],
            size=cluster.size(grid),
        )


@dataclasses.dataclass()
class ClusteringsLog:
    clusters: Dict[str, Dict[int, ClusterLog]]

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

                clusters[dtag_python][int(cluster_id_python)] = ClusterLog.from_cluster(cluster,
                                                                                        grid,
                                                                                        )

        return ClusteringsLog(clusters)


@dataclasses.dataclass()
class EventLog:
    dtag: str
    idx: int
    centroid: List[float]
    bdc: float
    size: int

    @staticmethod
    def from_event(event: Event, grid: Grid):
        return EventLog(
            dtag=event.event_id.dtag.dtag,
            idx=int(event.event_id.event_idx.event_idx),
            centroid=[float(x) for x in event.cluster.centroid],
            bdc=event.bdc.bdc,
            size=int(event.cluster.size(grid)),
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

            dtag_events[dtag][int(event_idx)] = EventLog.from_event(event, grid)

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

        return GridLog([float(x) for x in grid.shape()],
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
        return SiteLog(int(site_table_record.site_idx),
                       [float(x) for x in site_table_record.centroid],
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
    clusterings_merged: Dict[str, Dict[int, ClusterLog]]
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
            clusterings_merged={},
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
    exception: str

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
            exception="",
        )

    def print(self):
        pretty_printer = PrettyPrinter(indent=4, depth=1)

        log_dict = dataclasses.asdict(self)

        iterdict(log_dict)

        pretty_printer.pprint(log_dict)

    def save_json(self, path: Path):
        log_dict = dataclasses.asdict(self)

        remove_paths(log_dict)

        with open(str(path), "w") as f:
            json.dump(log_dict,
                      f,
                      )


def summarise_grid(grid: gemmi.FloatGrid):
    grid_array = np.array(grid, copy=False)

    summary = {
        f"Grid size": f"{grid.nu} {grid.nv} {grid.nw}",
        f"Grid spacegroup": f"{grid.spacegroup}",
        f"Grid unit cell": f"{grid.unit_cell}",
        f"Grid max": f"{np.max(grid_array)}",
        f"Grid min": f"{np.min(grid_array)}",
        f"Grid mean": f"{np.mean(grid_array)}",
    }

    return summary


def summarise_mtz(mtz: gemmi.Mtz):
    mtz_array = np.array(mtz, copy=False)

    summary = {

        f"Mtz shape": f"{mtz_array.shape}",
        f"Mtz spacegroup": f"{mtz.spacegroup}"
    }

    return summary


def summarise_structure(structure: gemmi.Structure):
    num_models: int = 0
    num_chains: int = 0
    num_residues: int = 0
    num_atoms: int = 0

    for model in structure:
        num_models += 1
        for chain in model:
            num_chains += 1
            for residue in chain:
                num_residues += 1
                for atom in residue:
                    num_atoms += 1

    summary = {
        f"Num models": f"{num_models}",
        f"Num chains": f"{num_chains}",
        f"Num residues": f"{num_residues}",
        f"Num atoms": f"{num_atoms}",
    }

    return summary


def summarise_event(event: Event):
    summary = {"Event system": f"{event.system}",
               f"Event dtag": "{event.dtag}",
               f"Event xyz": "{event.x} {event.y} {event.z}", }

    return summary


def summarise_array(array):
    summary = {
        "Shape": array.shape,
        "Mean": np.mean(array),
        "std": np.std(array),
        "min": np.min(array),
        "max": np.max(array),
    }
    return summary
