from __future__ import annotations
# from tkinter import Grid
from typing import *
from typing_extensions import ParamSpec, Concatenate, Self
from pathlib import Path
from enum import Enum

from numpy.typing import NDArray

import numpy as np
import gemmi

from enum import IntEnum


class Debug(IntEnum):
    DEFAULT = 0
    PRINT_SUMMARIES = 1
    PRINT_NUMERICS = 2
    AVERAGE_MAPS = 3
    DATASET_MAPS = 4
    INTERMEDIATE_FITS = 5


T = TypeVar('T', )
V = TypeVar("V")
P = ParamSpec("P")

NDArrayInterface = NDArray[np.float_]


# class NDArrayInterface(Protocol):
#     ...


class PanDDAConsoleInterface(Protocol):
    ...


class ProcessorInterface(Protocol):
    def __call__(self, funcs: Iterable[Callable[P, V]]) -> List[V]:
        ...


class ResidueIDInterface(Protocol):
    model: str
    chain: str
    insertion: str


class DtagInterface(Protocol):
    dtag: str


class DataDirInterface(Protocol):
    ...


DataDirsInterface = Dict[DtagInterface, DataDirInterface]


class AnalysesDirIntererface(Protocol):
    pandda_analyse_events_file: Path
    pandda_analyse_sites_file: Path


class ModelIDInterface(Protocol):
    model_id: int

    def __int__(self) -> int:
        ...


class ModelInterface(Protocol):
    mean: NDArrayInterface
    sigma_is: Dict[DtagInterface, float]
    sigma_s_m: NDArrayInterface


ModelsInterface = MutableMapping[ModelIDInterface, ModelInterface]


class XmapInterface(Protocol):
    xmap: CrystallographicGridInterface

    def to_array(self, copy=True) -> NDArrayInterface:
        ...


XmapsInterface = MutableMapping[DtagInterface, XmapInterface]


class DatasetModelsInterface(Protocol):
    path: Path


class ZMapFileInterface(Protocol):
    path: Path


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


class ProcessedDatasetsInterface(Protocol):
    path: Path
    processed_datasets: Dict[DtagInterface, ProcessedDatasetInterface]


class ShellDirInterface(Protocol):
    log_path: Path


# ShellDirsInterface = Dict[DtagInterface, ShellDirInterface]

class ShellDirsInterface(Protocol):
    path: Path
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
    events_json_file: Path
    tmp_dir: Path

    def build(self, get_dataset_smiles: GetDatasetSmilesInterface, process_local: Optional[ProcessorInterface] = None)\
            -> None:
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
    smoothing_factor: float

DatasetsInterface = Dict[DtagInterface, DatasetInterface]


class DatasetsStatisticsInterface(Protocol):
    ...


class AtomInterface(Protocol):
    ...


class StructureInterface(Protocol):
    def protein_atoms(self) -> Iterator[AtomInterface]:
        ...


class ResolutionInterface(Protocol):
    ...


class ReflectionsInterface(Protocol):
    def get_resolution(self) -> float:
        ...

    def transform_f_phi_to_map(self, f: str, phi: str, sample_rate: float) -> CrystallographicGridInterface:
        ...


class PositionInterface(Protocol):
    ...


class FractionalInterface(Protocol):
    ...


GridCoordInterface = Tuple[int, int, int]


class TransformInterface(Protocol):
    transform: gemmi.Transform
    com_moving: NDArrayInterface
    com_reference: NDArrayInterface

    def apply_reference_to_moving(self, alignment_positions: Dict[GridCoordInterface, PositionInterface]) -> \
            Dict[GridCoordInterface, PositionInterface]:
        ...

    def apply_moving_to_reference(
            self,
            alignment_positions: Dict[GridCoordInterface, PositionInterface],
    ) -> Dict[GridCoordInterface, PositionInterface]:
        ...


class PartitioningInterface(Protocol):
    inner_mask: CrystallographicGridInterface
    protein_mask: CrystallographicGridInterface
    total_mask: CrystallographicGridInterface
    symmetry_mask: CrystallographicGridInterface

    def __iter__(self) -> Iterator[ResidueIDInterface]:
        ...

    def __getitem__(self, residue_id: ResidueIDInterface) -> Dict[GridCoordInterface, PositionInterface]:
        ...


class UnitCellInterface(Protocol):
    def fractionalize(self, pos: PositionInterface) -> FractionalInterface:
        ...


class CrystallographicGridInterface(Protocol):
    nu: int
    nv: int
    nw: int
    unit_cell: UnitCellInterface

    def interpolate_value(self, grid_coord: FractionalInterface) -> float:
        ...


class GridInterface(Protocol):
    grid: CrystallographicGridInterface
    partitioning: PartitioningInterface

    def new_grid(self) -> CrystallographicGridInterface:
        ...


class AlignmentInterface(Protocol):
    def __iter__(self) -> Iterator[ResidueIDInterface]:
        ...

    def __getitem__(self, residue_id: ResidueIDInterface) -> TransformInterface:
        ...


AlignmentsInterface = Dict[DtagInterface, AlignmentInterface]


class ZmapInterface(Protocol):
    zmap: CrystallographicGridInterface

    def to_array(self, copy=True) -> NDArrayInterface:
        ...


ZmapsInterface = MutableMapping[DtagInterface, ZmapInterface]


class ClusterIDInterface(Protocol):
    def __int__(self) -> int:
        ...


class EDClusterInterface(Protocol):
    indexes: Tuple[NDArrayInterface]
    centroid: Tuple[float, float, float]
    values: NDArrayInterface
    cluster_positions_array: NDArrayInterface
    event_mask_indicies: Optional[Tuple[NDArrayInterface]]

    def size(self, grid: GridInterface) -> float:
        ...


class EDClusteringInterface(Protocol):
    clustering: MutableMapping[ClusterIDInterface, EDClusterInterface]

    def __len__(self) -> int:
        ...


EDClusteringsInterface = MutableMapping[DtagInterface, EDClusteringInterface]


class ModelSelectionInterface(Protocol):
    selected_model_id: ModelIDInterface
    log: Dict


class ModelResultInterface(Protocol):
    zmap: CrystallographicGridInterface
    clusterings: EDClusteringsInterface
    clusterings_large: EDClusteringsInterface
    clusterings_peaked: EDClusteringsInterface
    clusterings_merged: EDClusteringsInterface
    events: EventsInterface
    # event_scores: MutableMapping[EventIDInterface, float]
    event_scores: EventScoringResultsInterface
    model_log: Dict


ModelResultsInterface = MutableMapping[ModelIDInterface, ModelResultInterface]

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
    dtag: DtagInterface
    events: EventsInterface
    event_scores: EventScoringResultsInterface
    log: Any


DatasetResultsInterface = Dict[DtagInterface, DatasetResultInterface]


class ShellResultInterface(Protocol):
    shell: ShellInterface
    dataset_results: DatasetResultsInterface
    log: Dict


ShellResultsInterface = Dict[float, ShellResultInterface]


class BDCInterface(Protocol):
    bdc: float

    def __float__(self) -> float:
        return self.bdc


class EventInterface(Protocol):
    bdc: BDCInterface
    cluster: EDClusterInterface


# class EventRankingInterface(Protocol):
#     ...


class EventIDXInterface(Protocol):
    event_idx: int

    def __int__(self) -> int:
        ...


class EventIDInterface(Protocol):
    dtag: DtagInterface
    event_idx: EventIDXInterface


EventsInterface = Dict[EventIDInterface, EventInterface]

EventScoresInterface = MutableMapping[EventIDInterface, float]


class ConfromerIDInterface(Protocol):
    conformer_id: int

    def __int__(self):
        ...
        # return self.conformer_id

    def __hash__(self):
        ...


class ConformersInterface(Protocol):
    conformers: Dict[ConfromerIDInterface, Any]
    method: str
    path: Optional[Path]

    def log(self) -> Dict:
        ...


class ConformerFittingResultInterface(Protocol):
    score: Optional[float]
    optimised_fit: Optional[Any]
    score_log: Dict

    def log(self) -> Dict:
        ...


ConformerFittingResultsInterface = Dict[ConfromerIDInterface, ConformerFittingResultInterface]


class LigandFittingResultInterface(Protocol):
    conformer_fitting_results: ConformerFittingResultsInterface
    selected_conformer: ConfromerIDInterface

    def log(self) -> Dict:
        ...


class EventScoringResultInterface(Protocol):
    ligand_fitting_result: LigandFittingResultInterface

    def get_selected_structure(self) -> Any:
        ...

    def get_selected_structure_score(self) -> Optional[float]:
        ...

    def log(self) -> Dict:
        ...


EventScoringResultsInterface = Dict[EventIDInterface, EventScoringResultInterface]


class AutobuildResultInterface(Protocol):
    paths: List[str]
    scores: Dict[str, float]
    selected_fragment_path: Optional[str]

    def log(self) -> Any:
        ...


AutobuildResultsInterface = Dict[EventIDInterface, AutobuildResultInterface]


class SiteIDInterface(Protocol):
    site_id: int

    def __int__(self):
        return self.site_id


# SiteIDInterface = NewType(int)

class EventClasses(Enum):
    HIT = "hit"
    NEEDS_INSPECTION = "needs_inspection"
    JUNK = "junk"

    def __str__(self):
        return self.value


class EventClassificationInterface(Protocol):
    ...


EventClassificationsInterface = Dict[EventIDInterface, EventClasses]

EventRankingInterface = List[EventIDInterface]


class SiteInterface(Protocol):
    ...


class SitesInterface(Protocol):
    site_to_event: Dict[SiteIDInterface, List[EventIDInterface]]
    event_to_site: Dict[EventIDInterface, SiteIDInterface]
    centroids: Dict[SiteIDInterface, np.ndarray]


# SitesInterface = Dict[SiteIDInterface, SiteInterface]


class EventTableInterface(Protocol):
    def save(self, path: Path) -> None:
        ...


class SiteTableInterface(Protocol):
    def save(self, path: Path) -> None:
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
                 model_number: ModelIDInterface,
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
                 debug: Debug
                 ) -> ModelResultInterface:
        ...

class GetDatasetSmilesInterface(Protocol):
    def __call__(self, processed_dataset_path: Path,
                 output_smiles_path: Path,
                 ligand_pdb_path: Optional[Path],
                 ligand_cif_path: Optional[Path],
                 ligand_smiles_path: Optional[Path],
                 ) -> Optional[Path]:
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
                 ) -> DatasetsInterface:
        ...


class FilterIncompleteModelsInterface(FilterReferenceCompatibilityInterface, Protocol):
    def __call__(self,
                 datasets_diss_struc: DatasetsInterface,
                 reference: ReferenceInterface,
                 ) -> DatasetsInterface:
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
                 debug: Debug = Debug.DEFAULT,
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
                 debug: Debug = Debug.DEFAULT,
                 ):
        ...


class GetEDClusteringInterface(Protocol):
    def __call__(self, zmap: ZmapInterface, reference: ReferenceInterface, grid: GridInterface, contour_level: float,
                 cluster_cutoff_distance_multiplier: float) -> EDClusteringInterface:
        ...


# @runtime_checkable
class GetEventScoreInbuiltInterface(Protocol):
    tag: Literal["inbuilt"]

    def __call__(self,
                 test_dtag,
                 model_number,
                 dataset_processed_dataset,
                 dataset_xmap,
                 zmap,
                 events,
                 model,
                 grid,
                 dataset_alignment,
                 max_site_distance_cutoff,
                 min_bdc, max_bdc,
                 reference,
                 res, rate,
                 structure_output_folder,
                 debug: Debug
                 ) -> EventScoringResultsInterface:
        ...


# @runtime_checkable
class GetEventScoreAutobuildInterface(Protocol):
    tag: Literal["autobuild"]

    def __call__(self, *args, **kwargs) -> EventScoringResultsInterface:
        ...


# @runtime_checkable
class GetEventScoreSizeInterface(Protocol):
    tag: Literal["size"]

    def __call__(self,
                 events: EventsInterface
                 ) -> EventScoringResultsInterface:
        ...


GetEventScoreInterface = Union[
    GetEventScoreInbuiltInterface, GetEventScoreAutobuildInterface, GetEventScoreSizeInterface]


class GetAutobuildResultRhofitInterface(Protocol):
    tag: Literal["rhofit"] = "rhofit"

    def __call__(self,
                 dataset: DatasetInterface,
                 event: EventInterface,
                 pandda_fs: PanDDAFSModelInterface,
                 cif_strategy: str,
                 cut: float,
                 rhofit_coord: bool,
                 debug: Debug
                 ) -> AutobuildResultInterface:
        ...

class GetAutobuildResultInbuiltInterface(Protocol):
    tag: Literal["inbuilt"] = "inbuilt"
    def __call__(self, *args, **kwargs):
        ...

GetAutobuildResultInterface = Union[GetAutobuildResultRhofitInterface, GetAutobuildResultInbuiltInterface]

class GetSitesInterface(Protocol):
    def __call__(self,
                 all_events_ranked: Dict[EventIDInterface, EventInterface],
                 grid: GridInterface,
                 ) -> Dict[SiteIDInterface, SiteInterface]:
        ...


class GetSiteTableInterface(Protocol):
    def __call__(self,
                 events: EventsInterface,
                 sites: SitesInterface,
                 ) -> SiteTableInterface:
        ...


class GetEventTableInterface(Protocol):
    def __call__(self,
                 events: Dict[EventIDInterface, EventInterface],
                 event_ranking: EventRankingInterface,
                 ) -> EventTableInterface:
        ...


class GetEventClassTrivialInterface(Protocol):
    tag: Literal["trivial"] = "trivial"

    def __call__(self, event: EventInterface, ) -> EventClasses:
        ...


class GetEventClassAutobuildInterface(Protocol):
    tag: Literal["autobuild"] = "autobuild"

    def __call__(self, event: EventInterface, autobuild_result: AutobuildResultInterface) -> EventClasses:
        ...


GetEventClassInterface = Union[GetEventClassAutobuildInterface, GetEventClassTrivialInterface]


class GetEventRankingSizeInterface(Protocol):
    tag: Literal["size"] = "size"

    def __call__(self, events: EventsInterface, grid: GridInterface) -> EventRankingInterface:
        ...


class GetEventRankingAutobuildInterface(Protocol):
    tag: Literal["autobuild"] = "autobuild"

    def __call__(self,
                 events: EventsInterface,
                 autobuild_results: AutobuildResultsInterface,
                 datasets: DatasetsInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 ) -> EventRankingInterface:
        ...


class GetEventRankingSizeAutobuildInterface(Protocol):
    tag: Literal["size-autobuild"] = "size-autobuild"

    def __call__(self,
                 events: EventsInterface,
                 autobuild_results: AutobuildResultsInterface,
                 datasets: DatasetsInterface,
                 pandda_fs_model: PanDDAFSModelInterface,
                 ) -> EventRankingInterface:
        ...


GetEventRankingInterface = Union[
    GetEventRankingAutobuildInterface, GetEventRankingSizeInterface, GetEventRankingSizeAutobuildInterface
]
