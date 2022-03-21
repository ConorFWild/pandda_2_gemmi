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
    ...

class FilterRFreeInterface:
    ...

class FilterResolutionDatasets:
    ...


class FilterDissimilarModelsInterface:
    ...

class FilterDifferentSpacegroupsInterface:
    ...

class FilterIncompleteModelsInterface:
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
                 shell: ShellMultipleModels,
                 dataset_truncated_datasets,
                 alignments,
                 dataset_xmaps,
                 pandda_fs_model: PanDDAFSModel,
                 reference,
                 grid,
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