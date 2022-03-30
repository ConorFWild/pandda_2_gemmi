from __future__ import annotations
from typing import *
from typing_extensions import ParamSpec, Concatenate, Self
from pathlib import Path
from grpc import Call

import numpy as np

T = TypeVar('T',)
V = TypeVar("V", covariant=True)
P = ParamSpec("P")


# Analyse class interfaces
class PanDDAConsoleInterface(Protocol):
    ...


class ProcessorInterface(Protocol[P, V]):
    def __call__(self, funcs: Iterable[Callable[P, V]]) -> List[V]:
        ...


class DtagInterface(Protocol):
    dtag: str


class DataDirInterface(Protocol):
    ...

DataDirsInterface = Dict[DtagInterface, DataDirInterface]

class AnalysesDirIntererface(Protocol):
    ...


class ModelInterface(Protocol):
    ...


class XmapInterface(Protocol):
    ...


class DatasetModelsInterface(Protocol):
    path: Path

class ZMapFileInterface(Protocol):
    ...

class EventMapFilesInterface(Protocol):
    def add_event(self, event: EventInterface):
        ...

class LigandDirInterface(Protocol):
    ...


class ProcessedDatasetInterface(Protocol):
    path: Path
    dataset_models: DatasetModelsInterface
    input_mtz: Path
    input_pdb: Path
    source_mtz: Path
    source_pdb: Path
    z_map_file: ZMapFileInterface
    event_map_files: EventMapFilesInterface
    source_ligand_cif: Optional[Path]
    source_ligand_pdb: Optional[Path]
    source_ligand_smiles: Optional[Path]
    input_ligand_cif: Path
    input_ligand_pdb: Path
    input_ligand_smiles: Path
    source_ligand_dir: Optional[LigandDirInterface]
    input_ligand_dir: Path
    log_path: Path


ProcessedDatasetsInterface = Dict[DtagInterface, ProcessedDatasetInterface]


class ShellDirInterface(Protocol):
    ...

# ShellDirsInterface = Dict[DtagInterface, ShellDirInterface]

class ShellDirsInterface(Protocol):
    path : Path
    shell_dirs: Dict[float, ShellDirInterface]

    def build(self) -> None:
        ...


class PanDDAFSModelInterface(Protocol):
    # path: Path
    # processed_dataset_dirs: Dict[DtagInterface, ProcessedDatasetInterface]
    # shell_dirs: Dict[DtagInterface, ShellDirInterface]

    pandda_dir: Path
    data_dirs: DataDirsInterface
    analyses: AnalysesDirIntererface
    processed_datasets: ProcessedDatasetsInterface
    log_file: Path
    shell_dirs: Optional[ShellDirsInterface]
    console_log_file: Path

    def build(self) -> None:
        ...


    # def get_pandda_dir(self) -> Path:
    #     ...

    # def get_dataset_dir(self, dtag: DtagInterface) -> Path:
    #     ...

    # def get_dataset_structure_path(self, dtag: DtagInterface) -> Path:
    #     ...

    # def get_dataset_reflections_path(self, dtag: DtagInterface) -> Path:
    #     ...

    # def get_dataset_ligand_cif(self, dtag: DtagInterface) -> Optional[Path]:
    #     ...

    # def get_dataset_ligand_pdb(self, dtag: DtagInterface) -> Optional[Path]:
    #     ...

    # def get_dataset_ligand_smiles(self, dtag: DtagInterface) -> Optional[Path]:
    #     ...

    # def get_dataset_output_dir(self, dtag: DtagInterface) -> Path:
    #     ...

    # def get_event_map_path(self, dtag: DtagInterface) -> Path:
    #     ...

    # def get_shell_dir(self, shell_id: int) -> Path:
    #     ...

    # def get_event_table_path(self, ) -> Path:
    #     ...

    # def get_site_table_path(self, ) -> Path:
    #     ...

    # def get_log_path(self, ) -> Path:
    #     ...




class StructureFactorsInterface(Protocol):
    f: str
    phi: str


class ReferenceInterface(Protocol):
    dtag: DtagInterface
    dataset: DatasetInterface


class DatasetInterface(Protocol):
    structure: StructureInterface
    reflections: ReflectionsInterface


DatasetsInterface = Dict[DtagInterface, DatasetInterface]



class DatasetsStatisticsInterface(Protocol):
    ...

class StructureInterface(Protocol):
    ...


class ReflectionsInterface(Protocol):
    ...


class GridInterface(Protocol):
    ...


class AlignmentInterface(Protocol):
    ...


AlignmentsInterface = Dict[DtagInterface, AlignmentInterface]




class ModelResultInterface(Protocol):
    ...


ComparatorsInterface = MutableMapping[DtagInterface, MutableMapping[int, List[DtagInterface]]]


# def get_comparators(self) -> List[DtagInterface]:
#     ...


class ShellInterface(Protocol):
    res: float
    test_dtags: List[DtagInterface]
    train_dtags: Dict[int, List[DtagInterface]]
    all_dtags: List[DtagInterface]

    # def get_test_dtags(self) -> List[DtagInterface]:
    #     ...

    # def get_train_dtags(self, dtag: DtagInterface) -> List[DtagInterface]:
    #     ...

    # def get_all_dtags(self) -> List[DtagInterface]:
    #     ...


ShellsInterface = Dict[int, ShellInterface]

class DatasetResultInterface(Protocol):
    events: EventsInterface

DatasetResultsInterface = Dict[DtagInterface, DatasetResultInterface]

class ShellResultInterface(Protocol):
    shell: ShellInterface
    dataset_results: DatasetResultsInterface
    log: Dict


ShellResultsInterface = Dict[float, ShellResultInterface]


class EventInterface(Protocol):
    ...


class EventRankingInterface(Protocol):
    ...


class EventIDXInterface(Protocol):
    event_idx: int

class EventIDInterface(Protocol):
    dtag: DtagInterface
    event_idx: EventIDXInterface


EventsInterface = Dict[EventIDInterface, EventInterface]


class AutobuildResultInterface(Protocol):
    paths: List[str]
    scores: Dict[str, float]
    selected_fragment_path: Optional[str]

    def log(self) -> Any:
        ...

AutobuildResultsInterface = Dict[EventIDInterface, AutobuildResultInterface]


class SiteIDInterface(Protocol):
    ...


# SiteIDInterface = NewType(int)

class EventClassificationInterface(Protocol):
    ...


EventClassificationsInterface = Dict[EventIDInterface, EventClassificationInterface]


class SiteInterface(Protocol):
    ...


SitesInterface = Dict[SiteIDInterface, SiteInterface]


class EventTableInterface(Protocol):
    ...


class SiteTableInterface(Protocol):
    ...


# Ray
class RayCompatibleInterface(Protocol):
    def remote(self):
        ...


class PartialInterface(Protocol[P, V]):
    func: Callable[P, V]
    args: Any
    kwargs: Any

    def __init__(self, func: Callable[P, V], ):
        ...

    def paramaterise(self, 
        *args: P.args,
        **kwargs: P.kwargs,
        ) -> Self:
        ...

    def __call__(self) -> V:
        ...

#
# class PartialInterface2(Generic[P, V]):
#     func: Callable[P, V]
#     args: P.args
#     kwargs: P.kwargs
#
#     def __call__(self, *args: P.args) -> V:
#         ...
# #
# def decorator(f: Callable[P, int]) -> Callable[P, int]:
#     def wrapper(*args: P.args, **kw: P.kwargs) -> int:
#         print(args[0])
#         return 0
#     return wrapper

# PartialInterface = Callable[Concatenate[Callable[P, V], P], Callable[[], V]]

# Analyse Function Interfaces
class GetPanDDAFSModelInterface(Protocol):
    def __call__(self, 
    input_data_dirs: Path,
                 output_out_dir: Path,
                 pdb_regex: str, mtz_regex: str,
                 ligand_dir_name, ligand_cif_regex: str, ligand_pdb_regex: str, ligand_smiles_regex: str,
                 ) -> PanDDAFSModelInterface:
        ...

class GetDatasetsInterface(Protocol):
    def __call__(self, 
    pandda_fs_model: PanDDAFSModelInterface
                 ) -> DatasetsInterface:
        ...

class GetDatasetsStatisticsInterface(Protocol):
    def __call__(self, datasets: DatasetsInterface) -> DatasetsStatisticsInterface:
        ...
        


class SmoothBFactorsInterface(Protocol):
    def __call__(self,
                 dataset: DatasetInterface,
                 reference: ReferenceInterface,
                 structure_factors: StructureFactorsInterface,
                 ) -> DatasetInterface:
        ...


class LoadXMapInterface(Protocol):
    def __call__(self,
                 dataset: DatasetInterface,
                 alignment: AlignmentInterface,
                 grid: GridInterface,
                 structure_factors: StructureFactorsInterface,
                 sample_rate: float = 3.0,
                 ) -> XmapInterface:
        ...


class LoadXMapFlatInterface(Protocol):
    def __call__(self,
                 dataset: DatasetInterface,
                 alignment: AlignmentInterface,
                 grid: GridInterface,
                 structure_factors: StructureFactorsInterface,
                 sample_rate: float = 3.0,
                 ) -> np.ndarray:
        ...


class AnalyseModelInterface(Protocol):
    def __call__(self,
                 model: ModelInterface,
                 model_number: int,
                 test_dtag: DtagInterface,
                 dataset_xmap: XmapInterface,
                 reference: ReferenceInterface,
                 grid: GridInterface,
                 dataset_processed_dataset: ProcessedDatasetInterface,
                 dataset_alignment: AlignmentInterface,
                 max_site_distance_cutoff: float,
                 min_bdc: float,
                 max_bdc: float,
                 contour_level: float,
                 cluster_cutoff_distance_multiplier: float,
                 min_blob_volume: float,
                 min_blob_z_peak: float,
                 output_dir: Path,
                score_events_func: GetEventScoreInterface,
                 debug: bool
                 ) -> ModelResultInterface:
        ...


class DatasetsValidatorInterface(Protocol):
    def __call__(self, datasets: DatasetsInterface, exception: str):
        ...


class GetReferenceDatasetInterface(Protocol):
    def __call__(self, datasets: DatasetsInterface) -> ReferenceInterface:
        ...

class FilterBaseInterface(Protocol):

    def log(self) -> Dict[str, List[str]]:
        ...

    def exception(self) -> str:
        ...

    def name(self) -> str:
        ...


class FilterDataQualityInterface(FilterBaseInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: DatasetsInterface,
                 structure_factors: StructureFactorsInterface
                 ) -> DatasetsInterface:
        ...


class FiltersDataQualityInterface(Protocol):
    def __call__(self, datasets: DatasetsInterface, structure_factors: StructureFactorsInterface) -> DatasetsInterface:
        ...

    def log(self) -> Dict[str, List[str]]:
        ...


class FilterNoStructureFactorsInterface(FilterDataQualityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: DatasetsInterface,
                 structure_factors: StructureFactorsInterface
                 ) -> DatasetsInterface:
        ...


class FilterRFreeInterface(FilterDataQualityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: DatasetsInterface,
                 structure_factors: StructureFactorsInterface
                 ) -> DatasetsInterface:
        ...


class FilterResolutionDatasetsInterface(FilterDataQualityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: DatasetsInterface,
                 structure_factors: StructureFactorsInterface
                 ) -> DatasetsInterface:
        ...


class FilterReferenceCompatibilityInterface(FilterBaseInterface, Protocol, ):
    def __call__(self,
                 datasets_diss_struc: DatasetsInterface,
                 reference: ReferenceInterface
                 ) -> DatasetsInterface:
        ...


class FiltersReferenceCompatibilityInterface(Protocol):
    def __call__(self, datasets: DatasetsInterface, reference: ReferenceInterface) -> DatasetsInterface:
        ...

    def log(self) -> Dict[str, List[str]]:
        ...


class FilterDissimilarModelsInterface(FilterReferenceCompatibilityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: DatasetsInterface,
                 reference: ReferenceInterface,
                 ) -> DatasetsInterface:
        ...


class FilterDifferentSpacegroupsInterface(FilterReferenceCompatibilityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: DatasetsInterface,
                 reference: ReferenceInterface,
                 )-> DatasetsInterface:
        ...


class FilterIncompleteModelsInterface(FilterReferenceCompatibilityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: DatasetsInterface,
                 reference: ReferenceInterface,
                 )-> DatasetsInterface:
        ...


class DropReflectionColumns:
    def __call__(self, *args, **kwargs) -> DatasetsInterface:
        ...


class GetAlignmentsInterface(Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 ) -> Dict[DtagInterface, AlignmentInterface]:
        ...


class GetGridInterface(Protocol):
    def __call__(self,
                 reference: ReferenceInterface,
                 outer_mask: float,
                 inner_mask_symmetry: float,
                 sample_rate: float
                 ) -> GridInterface:
        ...


class GetComparatorsInterface(Protocol):
    def __call__(self,
                 datasets: Dict[DtagInterface, DatasetInterface],
                 alignments: Dict[DtagInterface, AlignmentInterface],
                 grid: GridInterface,
                 structure_factors: StructureFactorsInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 ) -> ComparatorsInterface:
        ...


class GetShellDirsInterface(Protocol):
    def __call__(self, pandda_dir: Path, shells: ShellsInterface) -> ShellDirsInterface:
        ...


class ProcessShellInterface(Protocol):
    def __call__(self,
                 shell: ShellInterface,
                 datasets: Dict[DtagInterface, DatasetInterface],
                 alignments: Dict[DtagInterface, AlignmentInterface],
                 grid: GridInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 reference: ReferenceInterface,
                 process_local: ProcessorInterface,
                 structure_factors: StructureFactorsInterface,
                 sample_rate: float,
                 contour_level: float,
                 cluster_cutoff_distance_multiplier: float,
                 min_blob_volume: float,
                 min_blob_z_peak: float,
                 outer_mask: float,
                 inner_mask_symmetry: float,
                 max_site_distance_cutoff: float,
                 min_bdc: float,
                 max_bdc: float,
                 memory_availability: str,
                 statmaps: bool,
                 load_xmap_func: LoadXMapInterface,
                 analyse_model_func: AnalyseModelInterface,
                 debug=False,
                 ):
        ...


class ProcessDatasetInterface(Protocol):
    def __call__(self,
                 test_dtag,
                 models,
                 shell: ShellInterface,
                 dataset_truncated_datasets,
                 alignments: Dict[DtagInterface, AlignmentInterface],
                 dataset_xmaps: Dict[DtagInterface, XmapInterface],
                 pandda_fs_model: PanDDAFSModelInterface,
                 reference: ReferenceInterface,
                 grid: GridInterface,
                 contour_level: float,
                 cluster_cutoff_distance_multiplier: float,
                 min_blob_volume: float,
                 min_blob_z_peak: float,
                 structure_factors: StructureFactorsInterface,
                 outer_mask: float,
                 inner_mask_symmetry: float,
                 max_site_distance_cutoff: float,
                 min_bdc: float,
                 max_bdc: float,
                 sample_rate: float,
                 statmaps: bool,
                 analyse_model_func: AnalyseModelInterface,
                 process_local=ProcessorInterface,
                 debug=bool,
                 ):
        ...


# @runtime_checkable
class GetEventScoreInbuiltInterface(Protocol):
    tag: Literal["inbuilt"]

    def __call__(self,
                 test_dtag,
                 model_number,
                 dataset_processed_dataset,
                 dataset_xmap,
                 events,
                 model,
                 grid,
                 dataset_alignment,
                 max_site_distance_cutoff,
                 min_bdc, max_bdc,
                 reference,
                 structure_output_folder,
                 debug
                 ) -> Dict[EventIDInterface, float]:
        ...


# @runtime_checkable
class GetEventScoreAutobuildInterface(Protocol):
    tag: Literal["autobuild"]

    def __call__(self, *args, **kwargs) -> Dict[EventIDInterface, float]:
        ...


GetEventScoreInterface = Union[GetEventScoreInbuiltInterface, GetEventScoreAutobuildInterface]


class GetAutobuildResultInterface(Protocol):
    def __call__(self,
                 dataset: DatasetInterface,
                 event: EventInterface,
                 pandda_fs: PanDDAFSModelInterface,
                 cif_strategy: str,
                 cut: float,
                 rhofit_coord: bool,
                 debug: bool
                 ) -> AutobuildResultInterface:
        ...


class GetEventRankingInterface(Protocol):
    def __call__(self,
                 all_events,
                 autobuild_results,
                 datasets,
                 pandda_fs_model,
                 ) -> EventRankingInterface:
        ...


class GetSitesInterface(Protocol):
    def __call__(self,
                 all_events_ranked: Dict[EventIDInterface, EventInterface],
                 grid: GridInterface,
                 ) -> Dict[SiteIDInterface, SiteInterface]:
        ...


class GetSiteTable(Protocol):
    def __call__(self,
                 sites: Dict[SiteIDInterface, SiteInterface]
                 ) -> SiteTableInterface:
        ...


class GetEventTableInterface(Protocol):
    def __call__(self,
                 events: Dict[EventIDInterface, EventInterface]
                 ) -> EventTableInterface:
        ...


class GetEventClassTrivialInterface(Protocol):
    tag: Literal["trivial"] = "trivial"
    def __call__(self, event: EventInterface, ) -> bool:
        ...


class GetEventClassAutobuildInterface(Protocol):
    tag: Literal["autobuild"] = "autobuild"
    def __call__(self, event: EventInterface, autobuild_result: AutobuildResultInterface) -> bool:
        ...

GetEventClassInterface = Union[GetEventClassAutobuildInterface, GetEventClassTrivialInterface]