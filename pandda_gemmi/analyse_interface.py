from __future__ import annotations
from typing import *
from pathlib import Path

T = TypeVar('T')


# Analyse class interfaces
class PanDDAConsoleInterface(Protocol):
    ...


class ProcessorInterface(Protocol):
    def __call__(self, funcs: List[Callable[..., T]]) -> List[T]:
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


class StructureInterface(Protocol):
    ...


class ReflectionsInterface(Protocol):
    ...


class GridInterface(Protocol):
    ...


class AlignmentInterface(Protocol):
    ...


class ModelInterface(Protocol):
    ...


class DtagInterface(Protocol):
    ...


class XmapInterface(Protocol):
    ...


class ProcessedDatasetInterface(Protocol):
    ...


class ModelResultInterface(Protocol):
    ...


class ComparatorsInterface(Protocol):
    ...


class ShellInterface(Protocol):
    ...


class EventInterface(Protocol):
    ...


class AutobuildResultInterface(Protocol):
    paths: List[str]
    scores: Dict[str, float]
    selected_fragment_path: Optional[str]


class EventRankingInterface(Protocol):
    ...


class EventIDInterface(Protocol):
    ...


class SiteIDInterface(Protocol):
    ...


class SiteInterface(Protocol):
    ...


class EventTableInterface(Protocol):
    ...


class SiteTableInterface(Protocol):
    ...


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


class FilterNoStructureFactorsInterface(Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 structure_factors: StructureFactorsInterface
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterRFreeInterface(Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 max_rfree: float
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterResolutionDatasets(Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 low_resolution_completeness: float
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterDissimilarModelsInterface(Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 max_rmsd_to_reference: float
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterDifferentSpacegroupsInterface(Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterIncompleteModelsInterface(Protocol):
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 ) -> Dict[DtagInterface, DatasetInterface]:
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
                 max_site_distance_cutoff: float,
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
