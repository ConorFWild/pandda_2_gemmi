from typing import *
from pathlib import Path


# Analyse class interfaces
class PanDDAConsoleInterface:
    ...


class ProcessorInterface:
    ...


class PanDDAFSModelInterface:
    ...


class StructureFactorsInterface:
    ...


class ReferenceInterface:
    ...


class DatasetInterface:
    ...


class GridInterface:
    ...


class AlignmentInterface:
    ...


class ModelInterface:
    ...


class DtagInterface:
    ...


class XmapInterface:
    ...


class ProcessedDatasetInterface:
    ...


class ModelResultInterface:
    ...


class ComparatorsInterface:
    ...


class ShellInterface:
    ...


class EventInterface:
    ...


class AutobuildResultInterface:
    ...


class EventRankingInterface:
    ...


class EventIDInterface:
    ...


class SiteIDInterface:
    ...


class SiteInterface:
    ...


class EventTableInterface:
    ...


class SiteTableInterface:
    ...


# Analyse Function Interfaces
class SmoothBFactorsInterface:
    def __call__(self,
                 dataset: DatasetInterface,
                 reference: ReferenceInterface,
                 structure_factors: StructureFactorsInterface,
                 ) -> DatasetInterface:
        ...


class LoadXMapInterface:
    def __call__(self,
                 dataset: DatasetInterface,
                 alignment: AlignmentInterface,
                 grid: GridInterface,
                 structure_factors: StructureFactorsInterface,
                 sample_rate: float = 3.0,
                 ) -> XmapInterface:
        ...


class AnalyseModelInterface:
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


class FilterNoStructureFactorsInterface:
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 structure_factors: StructureFactorsInterface
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterRFreeInterface:
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 max_rfree: float
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterResolutionDatasets:
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 low_resolution_completeness: float
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterDissimilarModelsInterface:
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 max_rmsd_to_reference: float
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterDifferentSpacegroupsInterface:
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class FilterIncompleteModelsInterface:
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 ) -> Dict[DtagInterface, DatasetInterface]:
        ...


class GetAlignmentsInterface:
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 ) -> Dict[DtagInterface, AlignmentInterface]:
        ...


class GetGridInterface:
    def __call__(self,
                 datasets_diss_struc: Dict[DtagInterface, DatasetInterface],
                 reference: ReferenceInterface,
                 outer_mask: float,
                 inner_mask_symmetry: float,
                 sample_rate: float
                 ) -> GridInterface:
        ...


class GetComparatorsInterface:
    def __call__(self,
                 datasets: Dict[DtagInterface, DatasetInterface],
                 alignments: Dict[DtagInterface, AlignmentInterface],
                 grid: GridInterface,
                 structure_factors: StructureFactorsInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 ):
        ...


class ProcessShellInterface:
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


class ProcessDatasetInterface:
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


class GetAutobuildResultInterface:
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


class GetEventRankingInterface:
    def __call__(self,
                 all_events,
                 autobuild_results,
                 datasets,
                 pandda_fs_model,
                 ) -> EventRankingInterface:
        ...


class GetSitesInterface:
    def __call__(self,
                 all_events_ranked: Dict[EventIDInterface, EventInterface],
                 grid: GridInterface,
                 max_site_distance_cutoff: float,
                 ) -> Dict[SiteIDInterface, SiteInterface]:
        ...


class GetSiteTable:
    def __call__(self,
                 sites: Dict[SiteIDInterface, SiteInterface]
                 ) -> SiteTableInterface:
        ...


class GetEventTableInterface:
    def __call__(self,
                 events: Dict[EventIDInterface, EventInterface]
                 ) -> EventTableInterface:
        ...

# GetPanDDAConsole = Callable[[], PanDDAConsoleInterface]
# GetProcessLocal
# GetProcessGlobal
#
# Smooth
# LoadXMap
# LoadXMapFlat
# AnalyseModel
# GetComparators
#
#
# get_smooth_func
# get_load_xmap_func
# get_load_xmap_flat_func
# get_analyse_model_func
