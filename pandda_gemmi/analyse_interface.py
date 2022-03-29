from __future__ import annotations
from typing import *
from typing_extensions import ParamSpec
from typing import ParamSpec
from pathlib import Path

import numpy as np

T = TypeVar('T')
V = TypeVar("V")
P = ParamSpec("P")


# Analyse class interfaces
class PanDDAConsoleInterface(Protocol):
    ...


class ProcessorInterface(Protocol):
    def __call__(self, funcs: List[Callable[P, T]]) -> List[T]:
        ...


class DtagInterface(Protocol):
    ...


class PanDDAFSModelInterface(Protocol):
    # path: Path
    # processed_dataset_dirs: Dict[DtagInterface, ProcessedDatasetInterface]
    # shell_dirs: Dict[DtagInterface, ShellDirInterface]

    def get_pandda_dir(self) -> Path:
        ...

    def get_dataset_dir(self, dtag: DtagInterface) -> Path:
        ...

    def get_dataset_structure_path(self, dtag: DtagInterface) -> Path:
        ...

    def get_dataset_reflections_path(self, dtag: DtagInterface) -> Path:
        ...

    def get_dataset_ligand_cif(self, dtag: DtagInterface) -> Optional[Path]:
        ...

    def get_dataset_ligand_pdb(self, dtag: DtagInterface) -> Optional[Path]:
        ...

    def get_dataset_ligand_smiles(self, dtag: DtagInterface) -> Optional[Path]:
        ...

    def get_dataset_output_dir(self, dtag: DtagInterface) -> Path:
        ...

    def get_event_map_path(self, dtag: DtagInterface) -> Path:
        ...

    def get_shell_dir(self, shell_id: int) -> Path:
        ...

    def get_event_table_path(self, ) -> Path:
        ...

    def get_site_table_path(self, ) -> Path:
        ...

    def get_log_path(self, ) -> Path:
        ...


class ShellDirInterface(Protocol):
    ...


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


class StructureInterface(Protocol):
    ...


class ReflectionsInterface(Protocol):
    ...


class GridInterface(Protocol):
    ...


class AlignmentInterface(Protocol):
    ...


AlignmentsInterface = Dict[DtagInterface, AlignmentInterface]


class ModelInterface(Protocol):
    ...


class XmapInterface(Protocol):
    ...


class ProcessedDatasetInterface(Protocol):
    ...


class ModelResultInterface(Protocol):
    ...


ComparatorsInterface = MutableMapping[DtagInterface, MutableMapping[int, List[DtagInterface]]]


# def get_comparators(self) -> List[DtagInterface]:
#     ...


class ShellInterface(Protocol):
    def get_test_dtags(self) -> List[DtagInterface]:
        ...

    def get_train_dtags(self, dtag: DtagInterface) -> List[DtagInterface]:
        ...

    def get_all_dtags(self) -> List[DtagInterface]:
        ...


ShellsInterface = Dict[int, ShellInterface]


class ShellResultInterface(Protocol):
    ...


ShellResultsInterface = Dict[int, ShellResultInterface]


class EventInterface(Protocol):
    ...


class EventRankingInterface(Protocol):
    ...


class EventIDInterface(Protocol):
    ...


EventsInterface = Dict[EventIDInterface, EventInterface]


class AutobuildResultInterface(Protocol):
    paths: List[str]
    scores: Dict[str, float]
    selected_fragment_path: Optional[str]


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
    args: P.args
    kwargs: P.kwargs


# Analyse Function Interfaces
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
                 debug: bool
                 ) -> ModelResultInterface:
        ...


class DatasetsValidatorInterface(Protocol):
    def __call__(self, datasets: DatasetsInterface, exception: str):
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
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 structure_factors: StructureFactorsInterface
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FiltersDataQualityInterface(Protocol):
    def __call__(self, datasets: DatasetsInterface, structure_factors: StructureFactorsInterface) -> DatasetsInterface:
        ...

    def log(self) -> Dict[str, List[str]]:
        ...


class FilterNoStructureFactorsInterface(FilterDataQualityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 structure_factors: StructureFactorsInterface
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterRFreeInterface(FilterDataQualityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 structure_factors: StructureFactorsInterface
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterResolutionDatasetsInterface(FilterDataQualityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 structure_factors: StructureFactorsInterface
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterReferenceCompatibilityInterface(FilterBaseInterface, Protocol, ):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FiltersReferenceCompatibilityInterface(Protocol):
    def __call__(self, datasets: DatasetsInterface, reference: ReferenceInterface) -> DatasetsInterface:
        ...

    def log(self) -> Dict[str, List[str]]:
        ...


class FilterDissimilarModelsInterface(FilterReferenceCompatibilityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterDifferentSpacegroupsInterface(FilterReferenceCompatibilityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterIncompleteModelsInterface(FilterReferenceCompatibilityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 ) -> Dict[DtagInterface, DatasetInterface]:
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
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
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
                 ):
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


class GetEventClassInterface(Protocol):
    def __call__(self, event: EventInterface, ) -> bool:
        ...


class GetEventClassAutobuildInterface(Protocol):
    def __call__(self, event: EventInterface, autobuild_result: AutobuildResultInterface) -> bool:
        ...
