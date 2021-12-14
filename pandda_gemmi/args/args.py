import dataclasses
from typing import *
import argparse
from pathlib import Path

from pandda_gemmi import constants


@dataclasses.dataclass()
class PanDDAArgs:
    data_dirs: Path
    out_dir: Path
    pdb_regex: str = "*.pdb"
    mtz_regex: str = "*.mtz"
    ligand_dir_regex: str = "compound"
    ligand_cif_regex: str = "*.cif"
    ligand_pdb_regex: str = "*.pdb"
    ligand_smiles_regex: str = "*.smiles"
    statmaps: bool = False
    low_memory: bool = False
    ground_state_datasets: Optional[List[str]] = None
    exclude_from_z_map_analysis: Optional[List[str]] = None
    exclude_from_characterisation: Optional[List[str]] = None
    only_datasets: Optional[List[str]] = None
    ignore_datasets: Optional[List[str]] = None
    dynamic_res_limits: bool = True
    high_res_upper_limit: float = 1.0
    high_res_lower_limit: float = 4.0
    high_res_increment: float = 0.05
    max_shell_datasets: int = 60
    min_characterisation_datasets: int = 15
    structure_factors: Optional[Tuple[str, str]] = None
    all_data_are_valid_values: bool = True
    low_resolution_completeness: float = 4.0
    sample_rate: float = 3.0
    max_rmsd_to_reference: float = 1.5
    max_rfree: float = 0.4
    max_wilson_plot_z_score: float = 1.5
    same_space_group_only: bool = False
    similar_models_only: bool = False
    resolution_factor: float = 0.25
    grid_spacing: float = 0.5
    padding: float = 3.0
    density_scaling: bool = True
    outer_mask: float = 8.0
    inner_mask: float = 2.0
    inner_mask_symmetry: float = 2.0
    contour_level: float = 2.5
    negative_values: bool = False
    min_blob_volume: float = 10.0
    min_blob_z_peak: float = 3.0
    clustering_cutoff: float = 1.5
    cluster_cutoff_distance_multiplier: float = 1.0
    max_site_distance_cutoff: float = 1.732
    min_bdc: float = 0.0
    max_bdc: float = 0.95
    increment: float = 0.05
    output_multiplier: float = 2.0
    comparison_strategy: str = "closest_cutoff"
    comparison_res_cutoff: float = 0.5
    comparison_min_comparators: int = 30
    comparison_max_comparators: int = 30
    known_apos: Optional[List[str]] = None
    exclude_local: int = 5
    cluster_selection: str = "close"
    local_processing: str = "multiprocessing_spawn"
    local_cpus: int = 12
    global_processing: str = "serial"
    memory_availability: str = "high"
    job_params_file: Optional[str] = None
    distributed_scheduler: str = "SGE"
    distributed_queue: str = "medium.q"
    distributed_project: str = "labxchem"
    distributed_num_workers: int = 12
    distributed_cores_per_worker: int = 12
    distributed_mem_per_core: int = 10
    distributed_resource_spec: str = "m_mem_free=10G"
    distributed_tmp: str = "/tmp"
    distributed_job_extra: str = ("--exclusive",)
    distributed_walltime: str = "30:00:00"
    distributed_watcher: bool = False
    distributed_slurm_partition: Optional[str] = None
    autobuild: bool = False
    autobuild_strategy: str = "rhofit"
    rhofit_coord: bool = False
    cif_strategy: str = "elbow"
    rank_method: str = "size"
    debug: bool = True

    @staticmethod
    def parse_only_datasets(string):
        if string:
            return string.split(",")
        else:
            return None

    @staticmethod
    def from_command_line():
        parser = argparse.ArgumentParser(
            description=constants.ARGS_DESCRIPTION,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # IO
        parser.add_argument(
            constants.ARGS_DATA_DIRS,
            type=Path,
            help=constants.ARGS_DATA_DIRS_HELP,
        )
        parser.add_argument(
            constants.ARGS_OUT_DIR,
            type=Path,
            help=constants.ARGS_OUT_DIR_HELP,
        )
        parser.add_argument(
            constants.ARGS_PDB_REGEX,
            type=str,
            help=constants.ARGS_PDB_REGEX_HELP,
        )
        parser.add_argument(
            constants.ARGS_MTZ_REGEX,
            type=str,
            help=constants.ARGS_MTZ_REGEX_HELP,
        )
        parser.add_argument(
            constants.ARGS_LIGAND_CIF_REGEX,
            type=str,
            default='*.cif',
            help=constants.ARGS_LIGAND_CIF_REGEX_HELP,
        )
        parser.add_argument(
            constants.ARGS_LIGAND_SMILES_REGEX,
            type=str,
            default='*.smiles',
            help=constants.ARGS_LIGAND_SMILES_REGEX_HELP,
        )
        parser.add_argument(
            constants.ARGS_LIGAND_PDB_REGEX,
            type=str,
            default='*.pdb',
            help=constants.ARGS_LIGAND_PDB_REGEX_HELP,
        )
        parser.add_argument(
            constants.ARGS_LIGAND_DIR_REGEX,
            type=str,
            default='compound',
            help=constants.ARGS_LIGAND_DIR_REGEX_HELP,
        )

        # Processing
        parser.add_argument(
            constants.ARGS_LOCAL_PROCESSING,
            type=str,
            default='multiprocessing_spawn',
            help=constants.ARGS_LOCAL_PROCESSING_HELP,
        )
        parser.add_argument(
            constants.ARGS_LOCAL_CPUS,
            type=int,
            default=12,
            help=constants.ARGS_LOCAL_CPUS_HELP,
        )
        parser.add_argument(
            constants.ARGS_GLOBAL_PROCESSING,
            type=str,
            default='serial',
            help=constants.ARGS_GLOBAL_PROCESSING_HELP,
        )
        parser.add_argument(
            constants.ARGS_MEMORY_AVAILABILITY,
            type=str,
            default='high',
            help=constants.ARGS_MEMORY_AVAILABILITY_HELP,
        )
        parser.add_argument(
            constants.ARGS_JOB_PARAMS_FILE,
            type=Path,
            default=None,
            help=constants.ARGS_JOB_PARAMS_FILE_HELP,
        )
        parser.add_argument(
            constants.ARGS_LOW_MEMORY,
            type=bool,
            default=False,
            help=constants.ARGS_LOW_MEMORY_HELP,
        )

        # Distribution
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_SCHEDULER,
            type=str,
            default='SGE',
            help=constants.ARGS_DISTRIBUTED_SCHEDULER_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_QUEUE,
            type=str,
            default='medium.q',
            help=constants.ARGS_DISTRIBUTED_QUEUE_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_PROJECT,
            type=str,
            default='labxchem',
            help=constants.ARGS_DISTRIBUTED_PROJECT_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_NUM_WORKERS,
            type=int,
            default=12,
            help=constants.ARGS_DISTRIBUTED_NUM_WORKERS_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_CORES_PER_WORKER,
            type=int,
            default=12,
            help=constants.ARGS_DISTRIBUTED_CORES_PER_WORKER_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_MEM_PER_CORE,
            type=int,
            default=10,
            help=constants.ARGS_DISTRIBUTED_MEM_PER_CORE_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_RESOURCE_SPEC,
            type=str,
            default='m_mem_free=10G',
            help=constants.ARGS_DISTRIBUTED_RESOURCE_SPEC_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_TMP,
            type=str,
            default='/tmp',
            help=constants.ARGS_DISTRIBUTED_TMP_HELP,
        )
        parser.add_argument(
            constants.ARGS_JOB_EXTRA,
            type=str,
            default='["--exclusive", ]',
            help=constants.ARGS_JOB_EXTRA_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_WALLTIME,
            type=str,
            default="30:00:00",
            help=constants.ARGS_DISTRIBUTED_WALLTIME_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_WATCHER,
            type=bool,
            default=False,
            help=constants.ARGS_DISTRIBUTED_WATCHER_HELP,
        )
        parser.add_argument(
            constants.ARGS_DISTRIBUTED_SLURM_PARTITION,
            type=bool,
            default=False,
            help=constants.ARGS_DISTRIBUTED_SLURM_PARTITION_HELP,
        )

        # Dataset Selection
        parser.add_argument(
            constants.ARGS_GROUND_STATE_DATASETS,
            type=list,
            default=None,
            help=constants.ARGS_GROUND_STATE_DATASETS_HELP,
        )
        parser.add_argument(
            constants.ARGS_EXCLUDE_FROM_Z_MAP_ANALYSIS,
            type=list,
            default=None,
            help=constants.ARGS_EXCLUDE_FROM_Z_MAP_ANALYSIS_HELP,
        )
        parser.add_argument(
            constants.ARGS_EXCLUDE_FROM_CHARACTERISATION,
            type=list,
            default=None,
            help=constants.ARGS_EXCLUDE_FROM_CHARACTERISATION_HELP,
        )
        parser.add_argument(
            constants.ARGS_ONLY_DATASETS,
            # type=list,
            default=None,
            help=constants.ARGS_ONLY_DATASETS_HELP,
        )
        parser.add_argument(
            constants.ARGS_IGNORE_DATASETS,
            type=list,
            default=None,
            help=constants.ARGS_IGNORE_DATASETS_HELP,
        )
        parser.add_argument(
            constants.ARGS_MAX_RMSD_TO_REFERENCE,
            type=float,
            default=1.5,
            help=constants.ARGS_MAX_RMSD_TO_REFERENCE_HELP,
        )
        parser.add_argument(
            constants.ARGS_MAX_RFREE,
            type=float,
            default=0.4,
            help=constants.ARGS_MAX_RFREE_HELP,
        )
        parser.add_argument(
            constants.ARGS_MAX_WILSON_PLOT_Z_SCORE,
            type=float,
            default=0.4,
            help=constants.ARGS_MAX_WILSON_PLOT_Z_SCORE_HELP,
        )
        parser.add_argument(
            constants.ARGS_SAME_SPACE_GROUP_ONLY,
            type=bool,
            default=False,
            help=constants.ARGS_SAME_SPACE_GROUP_ONLY_HELP,
        )
        parser.add_argument(
            constants.ARGS_SIMILAR_MODELS_ONLY,
            type=bool,
            default=False,
            help=constants.ARGS_SIMILAR_MODELS_ONLY_HELP,
        )

        # Comparator set finding actions

        parser.add_argument(
            constants.ARGS_COMPARISON_STRATEGY,
            type=str,
            default='closest_cutoff',
            help=constants.ARGS_COMPARISON_STRATEGY_HELP,
        )
        parser.add_argument(
            constants.ARGS_COMPARISON_RES_CUTOFF,
            type=float,
            default=0.5,
            help=constants.ARGS_DYNAMIC_RES_LIMITS_HELP,
        )
        parser.add_argument(
            constants.ARGS_COMPARISON_MIN_COMPARATORS,
            type=int,
            default=30,
            help=constants.ARGS_COMPARISON_MIN_COMPARATORS_HELP,
        )
        parser.add_argument(
            constants.ARGS_COMPARISON_MAX_COMPARATORS,
            type=int,
            default=30,
            help=constants.ARGS_COMPARISON_MAX_COMPARATORS_HELP,
        )
        parser.add_argument(
            constants.ARGS_KNOWN_APOS,
            type=list,
            default=None,
            help=constants.ARGS_KNOWN_APOS_HELP,
        )
        parser.add_argument(
            constants.ARGS_EXCLUDE_LOCAL,
            type=int,
            default=5,
            help=constants.ARGS_EXCLUDE_LOCAL_HELP,
        )
        parser.add_argument(
            constants.ARGS_CLUSTER_SELECTION,
            type=str,
            default='close',
            help=constants.ARGS_CLUSTER_SELECTION_HELP,
        )

        # Shell determination
        parser.add_argument(
            constants.ARGS_DYNAMIC_RES_LIMITS,
            type=bool,
            default=True,
            help=constants.ARGS_DYNAMIC_RES_LIMITS_HELP,
        )
        parser.add_argument(
            constants.ARGS_HIGH_RES_UPPER_LIMIT,
            type=float,
            default=1.0,
            help=constants.ARGS_HIGH_RES_UPPER_LIMIT_HELP,
        )
        parser.add_argument(
            constants.ARGS_HIGH_RES_LOWER_LIMIT,
            type=float,
            default=4.0,
            help=constants.ARGS_HIGH_RES_LOWER_LIMIT_HELP,
        )
        parser.add_argument(
            constants.ARGS_HIGH_RES_INCREMENT,
            type=float,
            default=0.05,
            help=constants.ARGS_HIGH_RES_INCREMENT_HELP,
        )
        parser.add_argument(
            constants.ARGS_MAX_SHELL_DATASETS,
            type=int,
            default=60,
            help=constants.ARGS_MAX_SHELL_DATASETS_HELP,
        )
        parser.add_argument(
            constants.ARGS_MIN_CHARACTERISATION_DATASETS,
            type=int,
            default=15,
            help=constants.ARGS_MIN_CHARACTERISATION_DATASETS_HELP,
        )

        # Diffraction data
        parser.add_argument(
            constants.ARGS_STRUCTURE_FACTORS,
            type=str,
            default=None,
            help=constants.ARGS_STRUCTURE_FACTORS_HELP,
        )
        parser.add_argument(
            constants.ARGS_ALL_DATA_ARE_VALID_VALUES,
            type=bool,
            default=True,
            help=constants.ARGS_ALL_DATA_ARE_VALID_VALUES_HELP,
        )
        parser.add_argument(
            constants.ARGS_LOW_RESOLUTION_COMPLETENESS,
            type=float,
            default=4.0,
            help=constants.ARGS_LOW_RESOLUTION_COMPLETENESS_HELP,
        )
        parser.add_argument(
            constants.ARGS_SAMPLE_RATE,
            type=float,
            default=3.0,
            help=constants.ARGS_SAMPLE_RATE_HELP,
        )

        # Map Options

        parser.add_argument(
            constants.ARGS_STATMAPS,
            type=bool,
            default=False,
            help=constants.ARGS_STATMAPS_HELP,
        )
        parser.add_argument(
            constants.ARGS_RESOLUTION_FACTOR,
            type=float,
            default=0.25,
            help=constants.ARGS_RESOLUTION_FACTOR_HELP,
        )
        parser.add_argument(
            constants.ARGS_GRID_SPACING,
            type=float,
            default=0.5,
            help=constants.ARGS_GRID_SPACING_HELP,
        )
        parser.add_argument(
            constants.ARGS_PADDING,
            type=float,
            default=3.0,
            help=constants.ARGS_PADDING_HELP,
        )
        parser.add_argument(
            constants.ARGS_DENSITY_SCALING,
            type=bool,
            default=True,
            help=constants.ARGS_DENSITY_SCALING_HELP,
        )
        parser.add_argument(
            constants.ARGS_OUTER_MASK,
            type=float,
            default=8.0,
            help=constants.ARGS_OUTER_MASK_HELP,
        )
        parser.add_argument(
            constants.ARGS_INNER_MASK,
            type=float,
            default=2.0,
            help=constants.ARGS_INNER_MASK_HELP,
        )
        parser.add_argument(
            constants.ARGS_INNER_MASK_SYMMETRY,
            type=float,
            default=2.0,
            help=constants.ARGS_INNER_MASK_SYMMETRY_HELP,
        )

        # ZMap CLustering options
        parser.add_argument(
            constants.ARGS_CONTOUR_LEVEL,
            type=float,
            default=2.5,
            help=constants.ARGS_CONTOUR_LEVEL_HELP,
        )
        parser.add_argument(
            constants.ARGS_NEGATIVE_VALUES,
            type=bool,
            default=False,
            help=constants.ARGS_NEGATIVE_VALUES_HELP,
        )
        parser.add_argument(
            constants.ARGS_MIN_BLOB_VOLUME,
            type=float,
            default=10.0,
            help=constants.ARGS_MIN_BLOB_VOLUME_HELP,
        )
        parser.add_argument(
            constants.ARGS_MIN_BLOB_Z_PEAK,
            type=float,
            default=3.0,
            help=constants.ARGS_MIN_BLOB_Z_PEAK_HELP,
        )
        parser.add_argument(
            constants.ARGS_CLUSTERING_CUTOFF,
            type=float,
            default=1.5,
            help=constants.ARGS_CLUSTERING_CUTOFF_HELP,
        )
        parser.add_argument(
            constants.ARGS_CLUSTER_CUTOFF_DISTANCE_MULTIPLIER,
            type=float,
            default=1.5,
            help=constants.ARGS_CLUSTER_CUTOFF_DISTANCE_MULTIPLIER_HELP,
        )

        # Site finding options
        parser.add_argument(
            constants.ARGS_MAX_SITE_DISTANCE_CUTOFF,
            type=float,
            default=1.732,
            help=constants.ARGS_MAX_SITE_DISTANCE_CUTOFF_HELP,
        )

        # BDC calculation options
        parser.add_argument(
            constants.ARGS_MIN_BDC,
            type=float,
            default=0.0,
            help=constants.ARGS_MIN_BDC_HELP,
        )
        parser.add_argument(
            constants.ARGS_MAX_BDC,
            type=float,
            default=0.95,
            help=constants.ARGS_MAX_BDC_HELP,
        )
        parser.add_argument(
            constants.ARGS_INCREMENT,
            type=float,
            default=0.05,
            help=constants.ARGS_INCREMENT_HELP,
        )
        parser.add_argument(
            constants.ARGS_OUTPUT_MULTIPLIER,
            type=float,
            default=2.0,
            help=constants.ARGS_OUTPUT_MULTIPLIER_HELP,
        )

        # Autobuilding
        parser.add_argument(
            constants.ARGS_AUTOBUILD,
            type=bool,
            default=False,
            help=constants.ARGS_AUTOBUILD_HELP,
        )
        parser.add_argument(
            constants.ARGS_AUTOBUILD_STRATEGY,
            type=str,
            default="rhofit",
            help=constants.ARGS_AUTOBUILD_STRATEGY_HELP,
        )
        parser.add_argument(
            constants.ARGS_RHOFIT_COORD,
            type=bool,
            default=False,
            help=constants.ARGS_RHOFIT_COORD_HELP,
        )
        parser.add_argument(
            constants.ARGS_CIF_STRATEGY,
            type=str,
            default='elbow',
            help=constants.ARGS_CIF_STRATEGY_HELP,
        )

        # Ranking
        parser.add_argument(
            constants.ARGS_RANK_METHOD,
            type=str,
            default='size',
            help=constants.ARGS_RANK_METHOD_HELP,
        )

        # Debug
        parser.add_argument(
            constants.ARGS_DEBUG,
            type=bool,
            default=False,
            help=constants.ARGS_DEBUG_HELP,
        )

        args = parser.parse_args()

        only_datasets = PanDDAArgs.parse_only_datasets(args.only_datasets)

        return PanDDAArgs(
            data_dirs=args.data_dirs,
            out_dir=args.out_dir,
            pdb_regex=args.pdb_regex,
            mtz_regex=args.mtz_regex,
            ligand_dir_regex=args.ligand_dir_regex,
            ligand_cif_regex=args.ligand_cif_regex,
            ligand_pdb_regex=args.ligand_pdb_regex,
            ligand_smiles_regex=args.ligand_smiles_regex,
            statmaps=args.statmaps,
            low_memory=args.low_memory,
            ground_state_datasets=args.ground_state_datasets,
            exclude_from_z_map_analysis=args.exclude_from_z_map_analysis,
            exclude_from_characterisation=args.exclude_from_characterisation,
            only_datasets=only_datasets,
            ignore_datasets=args.ignore_datasets,
            dynamic_res_limits=args.dynamic_res_limits,
            high_res_upper_limit=args.high_res_upper_limit,
            high_res_lower_limit=args.high_res_lower_limit,
            high_res_increment=args.high_res_increment,
            max_shell_datasets=args.max_shell_datasets,
            min_characterisation_datasets=args.min_characterisation_datasets,
            structure_factors=args.structure_factors,
            all_data_are_valid_values=args.all_data_are_valid_values,
            low_resolution_completeness=args.low_resolution_completeness,
            sample_rate=args.sample_rate,
            max_rmsd_to_reference=args.max_rmsd_to_reference,
            max_rfree=args.max_rfree,
            max_wilson_plot_z_score=args.max_wilson_plot_z_score,
            same_space_group_only=args.same_space_group_only,
            similar_models_only=args.similar_models_only,
            resolution_factor=args.resolution_factor,
            grid_spacing=args.grid_spacing,
            padding=args.padding,
            density_scaling=args.density_scaling,
            outer_mask=args.outer_mask,
            inner_mask=args.inner_mask,
            inner_mask_symmetry=args.inner_mask_symmetry,
            contour_level=args.contour_level,
            negative_values=args.negative_values,
            min_blob_volume=args.min_blob_volume,
            min_blob_z_peak=args.min_blob_z_peak,
            clustering_cutoff=args.clustering_cutoff,
            cluster_cutoff_distance_multiplier=args.cluster_cutoff_distance_multiplier,
            max_site_distance_cutoff=args.max_site_distance_cutoff,
            min_bdc=args.min_bdc,
            max_bdc=args.max_bdc,
            increment=args.increment,
            output_multiplier=args.output_multiplier,
            comparison_strategy=args.comparison_strategy,
            comparison_res_cutoff=args.comparison_res_cutoff,
            comparison_min_comparators=args.comparison_min_comparators,
            comparison_max_comparators=args.comparison_max_comparators,
            known_apos=args.known_apos,
            exclude_local=args.exclude_local,
            cluster_selection=args.cluster_selection,
            local_processing=args.local_processing,
            local_cpus=args.local_cpus,
            global_processing=args.global_processing,
            memory_availability=args.memory_availability,
            job_params_file=args.job_params_file,
            distributed_scheduler=args.distributed_scheduler,
            distributed_queue=args.distributed_queue,
            distributed_project=args.distributed_project,
            distributed_num_workers=args.distributed_num_workers,
            distributed_cores_per_worker=args.distributed_cores_per_worker,
            distributed_mem_per_core=args.distributed_mem_per_core,
            distributed_resource_spec=args.distributed_resource_spec,
            distributed_tmp=args.distributed_tmp,
            distributed_job_extra=args.job_extra,
            distributed_walltime=args.distributed_walltime,
            distributed_watcher=args.distributed_watcher,
            distributed_slurm_partition=args.distributed_slurm_partition,
            autobuild=args.autobuild,
            autobuild_strategy=args.autobuild_strategy,
            rhofit_coord=args.rhofit_coord,
            cif_strategy=args.cif_strategy,
            rank_method=args.rank_method,
            debug=args.debug,
        )
