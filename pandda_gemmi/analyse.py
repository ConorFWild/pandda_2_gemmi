# Base python
import os
import traceback
import time
import pprint
from functools import partial
import multiprocessing as mp
import json
import pickle

# Scientific python libraries
from dask.distributed import Client
import joblib

import ray

## Custom Imports
from pandda_gemmi import constants
from pandda_gemmi.common import Partial
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.dataset.dataset import GetReferenceDataset
from pandda_gemmi.edalignment.grid import GetGrid
from pandda_gemmi.pandda_logging import STDOUTManager, log_arguments, PanDDAConsole
from pandda_gemmi.dependencies import check_dependencies
from pandda_gemmi.dataset import (
    Datasets,
    Reference,
    StructureFactors,
    SmoothBFactors,
    DatasetsStatistics,
    drop_columns,
    GetDatasets,
    GetReferenceDataset,
)
from pandda_gemmi.edalignment import (Grid, GetGrid, Alignments, GetAlignments, from_unaligned_dataset_c,
                                      from_unaligned_dataset_c_flat, from_unaligned_dataset_c_ray,
                                      from_unaligned_dataset_c_flat_ray,
                                      LoadXmap, LoadXmapFlat
                                      )
from pandda_gemmi.filters import (
    FiltersDataQuality,
    FiltersReferenceCompatibility,
    FilterNoStructureFactors,
    FilterResolutionDatasets,
    FilterRFree,
    FilterDissimilarModels,
    FilterIncompleteModels,
    FilterDifferentSpacegroups,
    DatasetsValidator
)
from pandda_gemmi.comparators import (
    GetComparatorsHybrid, GetComparatorsHighResFirst, GetComparatorsHighResRandom, GetComparatorsHighRes,
    GetComparatorsCluster)
from pandda_gemmi.comparators import get_multiple_comparator_sets
from pandda_gemmi.shells import get_shells_multiple_models
from pandda_gemmi.logs import (
    save_json_log,
)

from pandda_gemmi.pandda_functions import (
    process_local_serial,
    process_local_joblib,
    process_local_multiprocessing,
    process_local_dask,
    process_local_ray,
    get_dask_client,
    process_global_serial,
    process_global_dask,
    get_shells,
    get_comparators_high_res_first,
    get_common_structure_factors,
)
from pandda_gemmi.event import Events, GetEventScoreInbuilt
from pandda_gemmi.ranking import (
    rank_events_size,
    rank_events_autobuild,
)
from pandda_gemmi.autobuild import (
    autobuild_rhofit,
    autobuild_rhofit_ray,
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
    GetAutobuildResultRhofit,
)
from pandda_gemmi.tables import (
    EventTable,
    SiteTable,
)
from pandda_gemmi.fs import PanDDAFSModel, ShellDirs, GetPanDDAFSModel, GetShellDirs
from pandda_gemmi.processing import (
    process_shell,
    process_shell_multiple_models,
    analyse_model,
    analyse_model_ray,
    RayWrapper,
    ProcessLocalRay
)

from pandda_gemmi import event_classification
from pandda_gemmi.sites import GetSites
from pandda_gemmi.analyse_interface import *

joblib.externals.loky.set_loky_pickler('pickle')
printer = pprint.PrettyPrinter()
console = PanDDAConsole()


def update_log(shell_log, shell_log_path):
    if shell_log_path.exists():
        os.remove(shell_log_path)

    with open(shell_log_path, "w") as f:
        json.dump(shell_log, f, indent=2)


def get_comparator_func(pandda_args: PanDDAArgs,
                        load_xmap_flat_func: LoadXMapFlatInterface,
                        process_local: ProcessorInterface) -> GetComparatorsInterface:
    if pandda_args.comparison_strategy == "closest":
        # Closest datasets after clustering
        raise NotImplementedError()

    elif pandda_args.comparison_strategy == "high_res":
        # Almost Old PanDDA strategy: highest res datasets

        comparators_func = GetComparatorsHighRes(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
        )

    elif pandda_args.comparison_strategy == "high_res_random":

        comparators_func = GetComparatorsHighResRandom(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
        )

    elif pandda_args.comparison_strategy == "high_res_first":
        # Old pandda strategy: random datasets that are higher resolution

        comparators_func = GetComparatorsHighResFirst(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
        )

    elif pandda_args.comparison_strategy == "cluster":
        comparators_func = GetComparatorsCluster(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
            sample_rate=pandda_args.sample_rate,
            # TODO: add option: pandda_args.resolution_cutoff,
            resolution_cutoff=3.0,
            load_xmap_flat_func=load_xmap_flat_func,
            process_local=process_local,
            debug=pandda_args.debug,
        )

    elif pandda_args.comparison_strategy == "hybrid":
        comparators_func = GetComparatorsHybrid(
            comparison_min_comparators=pandda_args.comparison_min_comparators,
            comparison_max_comparators=pandda_args.comparison_max_comparators,
            sample_rate=pandda_args.sample_rate,
            # TODO: add option: pandda_args.resolution_cutoff,
            resolution_cutoff=3.0,
            load_xmap_flat_func=load_xmap_flat_func,
            process_local=process_local,
            debug=pandda_args.debug,
        )

    else:
        raise Exception("Unrecognised comparison strategy")

    return comparators_func


def get_process_global(pandda_args, distributed_tmp):
    if pandda_args.global_processing == "serial":
        process_global = process_global_serial
    elif pandda_args.global_processing == "distributed":
        client = get_dask_client(
            scheduler=pandda_args.distributed_scheduler,
            num_workers=pandda_args.distributed_num_workers,
            queue=pandda_args.distributed_queue,
            project=pandda_args.distributed_project,
            cores_per_worker=pandda_args.local_cpus,
            distributed_mem_per_core=pandda_args.distributed_mem_per_core,
            resource_spec=pandda_args.distributed_resource_spec,
            job_extra=pandda_args.distributed_job_extra,
            walltime=pandda_args.distributed_walltime,
            watcher=pandda_args.distributed_watcher,
        )
        process_global = partial(
            process_global_dask,
            client=client,
            tmp_dir=distributed_tmp
        )
    else:
        raise Exception(f"Could not find an implementation of --global_processing: {pandda_args.global_processing}")

    return process_global


def get_process_local(pandda_args):
    if pandda_args.local_processing == "serial":
        process_local = process_local_serial

    elif pandda_args.local_processing == "joblib":
        process_local = partial(process_local_joblib, n_jobs=pandda_args.local_cpus, verbose=50, max_nbytes=None)

    elif pandda_args.local_processing == "multiprocessing_forkserver":
        mp.set_start_method("forkserver")
        process_local = partial(process_local_multiprocessing, n_jobs=pandda_args.local_cpus, method="forkserver")
        # process_local_load = partial(process_local_joblib, int(joblib.cpu_count() * 3), "threads")

    elif pandda_args.local_processing == "multiprocessing_spawn":
        mp.set_start_method("spawn")
        process_local = partial(process_local_multiprocessing, n_jobs=pandda_args.local_cpus, method="spawn")
        # process_local_load = partial(process_local_joblib, int(joblib.cpu_count() * 3), "threads")
    elif pandda_args.local_processing == "dask":
        client = Client(n_workers=pandda_args.local_cpus)
        process_local = partial(
            process_local_dask,
            client=client
        )

    elif pandda_args.local_processing == "ray":
        ray.init(num_cpus=pandda_args.local_cpus)
        process_local = ProcessLocalRay()

    else:
        raise Exception()

    return process_local


def get_smooth_func(pandda_args: PanDDAArgs) -> SmoothBFactorsInterface:
    smooth_func: SmoothBFactorsInterface = SmoothBFactors()

    return smooth_func


def get_load_xmap_func(pandda_args) -> LoadXMapInterface:
    load_xmap_func = LoadXmap()
    return load_xmap_func


def get_load_xmap_flat_func(pandda_args) -> LoadXMapFlatInterface:
    load_xmap_flat_func = LoadXmapFlat()
    return load_xmap_flat_func


def get_analyse_model_func(pandda_args):
    analyse_model_func = analyse_model
    return analyse_model_func


def get_filter_data_quality(
        filter_keys: List[str],
        datasets_validator: DatasetsValidatorInterface,
        pandda_args: PanDDAArgs,
) -> FiltersDataQualityInterface:
    filters = {}

    if "structure_factors" in filter_keys:
        filters["structure_factors"] = FilterNoStructureFactors()

    if "resolution" in filter_keys:
        filters["resolution"] = FilterResolutionDatasets(pandda_args.low_resolution_completeness)

    if "rfree" in filter_keys:
        filters["rfree"] = FilterRFree(pandda_args.max_rfree)

    return FiltersDataQuality(filters, datasets_validator)


def get_filter_reference_compatability(
        filter_keys: List[str],
        datasets_validator: DatasetsValidatorInterface,
        pandda_args: PanDDAArgs,
) -> FiltersReferenceCompatibilityInterface:
    filters = {}

    if "dissimilar_models" in filter_keys:
        filters["dissimilar_models"] = FilterDissimilarModels(pandda_args.max_rmsd_to_reference)

    if "large_gaps" in filter_keys:
        filters["large_gaps"] = FilterIncompleteModels()

    if "dissimilar_spacegroups" in filter_keys:
        filters["dissimilar_spacegroups"] = FilterDifferentSpacegroups()

    return FiltersReferenceCompatibility(filters, datasets_validator)


def get_score_events_func(pandda_args: PanDDAArgs) -> GetEventScoreInterface:
    return GetEventScoreInbuilt()


def process_pandda(pandda_args: PanDDAArgs, ):
    ###################################################################
    # # Configuration
    ###################################################################
    time_start = time.time()

    # Process args
    distributed_tmp = Path(pandda_args.distributed_tmp)

    # CHeck dependencies
    # with STDOUTManager('Checking dependencies...', '\tAll dependencies validated!'):
    console.start_dependancy_check()
    check_dependencies(pandda_args)

    # Initialise log
    # with STDOUTManager('Initialising log...', '\tPanDDA log initialised!'):
    console.start_log()
    pandda_log: Dict = {}
    pandda_log[constants.LOG_START] = time.time()
    initial_args = log_arguments(pandda_args, )

    pandda_log[constants.LOG_ARGUMENTS] = initial_args

    # Get global processor
    console.start_initialise_shell_processor()
    process_global: ProcessorInterface = get_process_global(pandda_args, distributed_tmp)

    # Get local processor
    console.start_initialise_multiprocessor()
    process_local: ProcessorInterface = get_process_local(pandda_args)

    # Get reflection smoothing function
    smooth_func: SmoothBFactorsInterface = get_smooth_func(pandda_args)

    # Get XMap loading functions
    load_xmap_func: LoadXMapInterface = get_load_xmap_func(pandda_args)
    load_xmap_flat_func: LoadXMapFlatInterface = get_load_xmap_flat_func(pandda_args)

    # Get the filtering functions
    datasets_validator: DatasetsValidatorInterface = DatasetsValidator(pandda_args.min_characterisation_datasets)
    filter_data_quality: FiltersDataQualityInterface = get_filter_data_quality(
        [
            "structure_factors",
            "resolution",
            "rfree"
        ],
        datasets_validator,
        pandda_args
    )
    filter_reference_compatability: FiltersReferenceCompatibilityInterface = get_filter_reference_compatability(
        [
            "dissimilar_models",
            "large_gaps"
            "dissimilar_spacegroups"
        ],
        datasets_validator,
        pandda_args
    )

    # Get the functino for selecting the comparators
    comparators_func: GetComparatorsInterface = get_comparator_func(
        pandda_args,
        load_xmap_flat_func,
        process_local
    )

    # Get the staticial model anaylsis function
    analyse_model_func: AnalyseModelInterface = get_analyse_model_func(pandda_args)

    # Get the event scoring func
    score_events_func: GetEventScoreInterface = get_score_events_func(pandda_args)

    # Set up autobuilding function
    autobuild_func: Optional[GetAutobuildResultInterface] = None
    if pandda_args.autobuild:

        # with STDOUTManager('Setting up autobuilding...', '\tSet up autobuilding!'):
        if pandda_args.autobuild_strategy == "rhofit":
            autobuild_func: Optional[GetAutobuildResultInterface] = GetAutobuildResultRhofit()

        elif pandda_args.autobuild_strategy == "inbuilt":
            raise NotImplementedError("Autobuilding with inbuilt method is not yet implemented")


        else:
            raise Exception(f"Autobuild strategy: {pandda_args.autobuild_strategy} is not valid!")

    # Get the event classification function
    if pandda_args.autobuild:
        get_event_class: GetEventClassInterface = event_classification.GetEventClassAutobuildScore(0.4)
    else:
        get_event_class: GetEventClassInterface = event_classification.GetEventClassTrivial()

    # Get the site determination function
    get_sites: GetSitesInterface = GetSites(pandda_args.max_site_distance_cutoff)

    ###################################################################
    # # Run the PanDDA
    ###################################################################
    try:

        ###################################################################
        # # Get datasets
        ###################################################################
        console.start_fs_model()
        time_fs_model_building_start = time.time()
        pandda_fs_model: PanDDAFSModelInterface = GetPanDDAFSModel()(
            pandda_args.data_dirs,
            pandda_args.out_dir,
            pandda_args.pdb_regex,
            pandda_args.mtz_regex,
            pandda_args.ligand_dir_regex,
            pandda_args.ligand_cif_regex,
            pandda_args.ligand_pdb_regex,
            pandda_args.ligand_smiles_regex,
        )
        pandda_fs_model.build()
        time_fs_model_building_finish = time.time()
        pandda_log["FS model building time"] = time_fs_model_building_finish - time_fs_model_building_start

        console.summarise_fs_model(pandda_fs_model)
        update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

        ###################################################################
        # # Load datasets
        ###################################################################
        # Get datasets
        console.start_load_datasets()
        datasets_initial: DatasetsInterface = GetDatasets()(pandda_fs_model, )
        datasets_statistics: DatasetsStatisticsInterface = DatasetsStatistics(datasets_initial)
        console.summarise_datasets(datasets_initial, datasets_statistics)

        # If structure factors not given, check if any common ones are available
        with STDOUTManager('Looking for common structure factors in datasets...', f'\tFound structure factors!'):
            if not pandda_args.structure_factors:
                potential_structure_factors: Optional[StructureFactorsInterface] = get_common_structure_factors(
                    datasets_initial)
                # If still no structure factors
                if not potential_structure_factors:
                    raise Exception(
                        "No common structure factors found in mtzs. Please manually provide the labels with the --structure_factors option.")
                else:
                    structure_factors: StructureFactorsInterface = potential_structure_factors
            else:
                structure_factors: StructureFactorsInterface = StructureFactors(pandda_args.structure_factors[0],
                                                                                pandda_args.structure_factors[1])

        ###################################################################
        # # Data Quality filters
        ###################################################################
        console.start_data_quality_filters()

        datasets_for_filtering: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                                     datasets_initial.items()}

        datasets_quality_filtered: DatasetsInterface = filter_data_quality(datasets_for_filtering, structure_factors)

        ###################################################################
        # # Truncate columns
        ###################################################################
        datasets_wilson: DatasetsInterface = drop_columns(datasets_quality_filtered, structure_factors)

        ###################################################################
        # # Reference Selection
        ###################################################################
        console.start_reference_selection()

        # Select refernce
        with STDOUTManager('Deciding on reference dataset...', f'\tDone!'):
            reference: ReferenceInterface = GetReferenceDataset()(
                datasets_wilson, 
                datasets_statistics,
                )
            pandda_log["Reference Dtag"] = str(reference.dtag)
            console.summarise_reference(reference)
            if pandda_args.debug:
                print(reference.dtag)

            if pandda_args.debug:
                with open(pandda_fs_model.pandda_dir / "reference.pickle", "wb") as f:
                    pickle.dump(reference, f)

        ###################################################################
        # # B Factor smoothing
        ###################################################################
        console.start_b_factor_smoothing()

        with STDOUTManager('Performing b-factor smoothing...', f'\tDone!'):
            start = time.time()
            datasets_smoother: DatasetsInterface = {
                smoothed_dtag: smoothed_dataset
                for smoothed_dtag, smoothed_dataset
                in zip(
                    datasets_wilson,
                    process_local(
                        [
                            Partial(smooth_func).paramaterise(
                                dataset,
                                reference,
                                structure_factors,
                            )
                            for dtag, dataset
                            in datasets_wilson.items()
                        ]
                    )
                )
            }

            finish = time.time()
            pandda_log["Time to perform b factor smoothing"] = finish - start

        ###################################################################
        # # Reference compatability filters
        ###################################################################
        console.start_reference_comparability_filters()

        datasets_reference: DatasetsInterface = filter_reference_compatability(datasets_smoother, reference)
        datasets: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                       datasets_reference.items()}

        ###################################################################
        # # Getting grid
        ###################################################################
        console.start_get_grid()

        # Grid
        with STDOUTManager('Getting the analysis grid...', f'\tDone!'):
            grid: GridInterface = GetGrid()(reference,
                                                      pandda_args.outer_mask,
                                                      pandda_args.inner_mask_symmetry,
                                                      # sample_rate=pandda_args.sample_rate,
                                                      sample_rate=reference.dataset.reflections.resolution().resolution / 0.5
                                                      )

            if pandda_args.debug:
                with open(pandda_fs_model.pandda_dir / "grid.pickle", "wb") as f:
                    pickle.dump(grid, f)

        ###################################################################
        # # Getting alignments
        ###################################################################

        console.start_alignments()

        with STDOUTManager('Getting local alignments of the electron density to the reference...', f'\tDone!'):
            alignments: AlignmentsInterface = GetAlignments()(
                reference,
                datasets,
            )

            if pandda_args.debug:
                with open(pandda_fs_model.pandda_dir / "alighments.pickle", "wb") as f:
                    pickle.dump(alignments, f)

        update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

        ###################################################################
        # # Assign comparison datasets
        ###################################################################
        console.start_get_comparators()

        with STDOUTManager('Deciding on the datasets to characterise the groundstate for each dataset to analyse...',
                           f'\tDone!'):
            # TODO: Fix typing for comparators func
            comparators: ComparatorsInterface = comparators_func(
                datasets,
                alignments,
                grid,
                structure_factors,
                pandda_fs_model,
            )

        # if pandda_args.comparison_strategy == "cluster" or pandda_args.comparison_strategy == "hybrid":
        #     pandda_log["Cluster Assignments"] = {str(dtag): int(cluster) for dtag, cluster in
        #                                          cluster_assignments.items()}
        #     pandda_log["Neighbourhood core dtags"] = {int(neighbourhood_number): [str(dtag) for dtag in
        #                                                                           neighbourhood.core_dtags]
        #                                               for neighbourhood_number, neighbourhood
        #                                               in comparators.items()
        #                                               }

        # if pandda_args.debug:
        #     print("Comparators are:")
        # printer.pprint(pandda_log["Cluster Assignments"])
        # printer.pprint(pandda_log["Neighbourhood core dtags"])
        # printer.pprint(comparators)

        update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

        ###################################################################
        # # Process shells
        ###################################################################
        console.start_process_shells()

        # Partition the Analysis into shells in which all datasets are being processed at a similar resolution for the
        # sake of computational efficiency
        with STDOUTManager('Deciding on how to partition the datasets into resolution shells for processing...',
                           f'\tDone!'):
            if pandda_args.comparison_strategy == "cluster" or pandda_args.comparison_strategy == "hybrid":
                shells: ShellsInterface = get_shells_multiple_models(
                    datasets,
                    comparators,
                    pandda_args.min_characterisation_datasets,
                    pandda_args.max_shell_datasets,
                    pandda_args.high_res_increment,
                    pandda_args.only_datasets,
                    debug=pandda_args.debug,
                )
                if pandda_args.debug:
                    print('Got shells that support multiple models')
                    for shell_res, shell in shells.items():
                        print(f'\tShell res: {shell.res}: {shell.test_dtags[:3]}')
                        for cluster_num, dtags in shell.train_dtags.items():
                            print(f'\t\t{cluster_num}: {dtags[:5]}')

            else:
                shells: ShellsInterface = get_shells(
                    datasets,
                    comparators,
                    pandda_args.min_characterisation_datasets,
                    pandda_args.max_shell_datasets,
                    pandda_args.high_res_increment,
                    pandda_args.only_datasets,

                )
            pandda_fs_model.shell_dirs = GetShellDirs()(pandda_fs_model.pandda_dir, shells)
            pandda_fs_model.shell_dirs.build()

        if pandda_args.debug:
            printer.pprint(shells)

        # Process the shells
        with STDOUTManager('Processing the shells...', f'\tDone!'):
            time_shells_start = time.time()
            if pandda_args.comparison_strategy == "cluster" or pandda_args.comparison_strategy == "hybrid":

                shell_results: ShellResultsInterface = {
                    shell_id: shell_result
                    for shell_id, shell_result
                    in zip(
                        shells,
                        process_global(
                            [
                                Partial(
                                    process_shell_multiple_models).paramaterise(
                                    shell,
                                    datasets,
                                    alignments,
                                    grid,
                                    pandda_fs_model,
                                    reference,
                                    process_local=process_local,
                                    structure_factors=structure_factors,
                                    sample_rate=pandda_args.sample_rate,
                                    contour_level=pandda_args.contour_level,
                                    cluster_cutoff_distance_multiplier=pandda_args.cluster_cutoff_distance_multiplier,
                                    min_blob_volume=pandda_args.min_blob_volume,
                                    min_blob_z_peak=pandda_args.min_blob_z_peak,
                                    outer_mask=pandda_args.outer_mask,
                                    inner_mask_symmetry=pandda_args.inner_mask_symmetry,
                                    max_site_distance_cutoff=pandda_args.max_site_distance_cutoff,
                                    min_bdc=pandda_args.min_bdc,
                                    max_bdc=pandda_args.max_bdc,
                                    memory_availability=pandda_args.memory_availability,
                                    statmaps=pandda_args.statmaps,
                                    load_xmap_func=load_xmap_func,
                                    analyse_model_func=analyse_model_func,
                                    score_events_func=score_events_func,
                                    debug=pandda_args.debug,
                                )
                                for res, shell
                                in shells.items()
                            ],
                        )
                    )
                }

            else:
                shell_results: ShellResultsInterface = {
                    shell_id: shell_result
                    for shell_id, shell_result
                    in zip(
                        shells,
                        process_global(
                            [
                                Partial(process_shell).paramaterise(
                                    shell,
                                    datasets,
                                    alignments,
                                    grid,
                                    pandda_fs_model,
                                    reference,
                                    process_local=process_local,
                                    structure_factors=structure_factors,
                                    sample_rate=pandda_args.sample_rate,
                                    contour_level=pandda_args.contour_level,
                                    cluster_cutoff_distance_multiplier=pandda_args.cluster_cutoff_distance_multiplier,
                                    min_blob_volume=pandda_args.min_blob_volume,
                                    min_blob_z_peak=pandda_args.min_blob_z_peak,
                                    outer_mask=pandda_args.outer_mask,
                                    inner_mask_symmetry=pandda_args.inner_mask_symmetry,
                                    max_site_distance_cutoff=pandda_args.max_site_distance_cutoff,
                                    min_bdc=pandda_args.min_bdc,
                                    max_bdc=pandda_args.max_bdc,
                                    memory_availability=pandda_args.memory_availability,
                                    statmaps=pandda_args.statmaps,
                                    load_xmap_func=load_xmap_func,
                                )
                                for res, shell
                                in shells.items()
                            ],
                        )
                    )
                }

            time_shells_finish = time.time()
            pandda_log[constants.LOG_SHELLS] = {
                res: shell_result.log
                for res, shell_result
                in shell_results.items()
                if shell_result
            }
            pandda_log["Time to process all shells"] = time_shells_finish - time_shells_start
            if pandda_args.debug:
                print(f"Time to process all shells: {time_shells_finish - time_shells_start}")

        all_events: EventsInterface = {}
        for res, shell_result in shell_results.items():
            if shell_result:
                for dtag, dataset_result in shell_result.dataset_results.items():
                    all_events.update(dataset_result.events)

        # Add the event maps to the fs
        for event_id, event in all_events.items():
            pandda_fs_model.processed_datasets[event_id.dtag].event_map_files.add_event(event)

        update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

        ###################################################################
        # # Autobuilding
        ###################################################################
        # Autobuild the results if set to
        autobuild_results = {}
        if autobuild_func:
            console.start_autobuilding()

            with STDOUTManager('Attempting to autobuild events...', f'\tDone!'):

                if pandda_args.global_processing == 'serial':
                    process_autobuilds = process_local
                else:
                    process_autobuilds = process_global

                time_autobuild_start = time.time()
                autobuild_results: AutobuildResultsInterface = {
                    event_id: autobuild_result
                    for event_id, autobuild_result
                    in zip(
                        all_events,
                        process_autobuilds(
                            [
                                Partial(autobuild_func).paramaterise(
                                    datasets[event_id.dtag],
                                    all_events[event_id],
                                    pandda_fs_model,
                                    cif_strategy=pandda_args.cif_strategy,
                                    cut=2.0,
                                    rhofit_coord=pandda_args.rhofit_coord,
                                    debug=pandda_args.debug,
                                )
                                for event_id
                                in all_events
                            ],
                        )
                    )
                }

                time_autobuild_finish = time.time()
                pandda_log[constants.LOG_AUTOBUILD_TIME] = time_autobuild_finish - time_autobuild_start

                # Save results
                pandda_log[constants.LOG_AUTOBUILD_COMMANDS] = {}
                for event_id, autobuild_result in autobuild_results.items():
                    dtag = str(event_id.dtag)
                    if dtag not in pandda_log[constants.LOG_AUTOBUILD_COMMANDS]:
                        pandda_log[constants.LOG_AUTOBUILD_COMMANDS][dtag] = {}

                    event_idx = int(event_id.event_idx.event_idx)

                    pandda_log[constants.LOG_AUTOBUILD_COMMANDS][dtag][event_idx] = autobuild_result.log()

            with STDOUTManager('Updating the PanDDA models with best scoring fragment build...', f'\tDone!'):
                # Add the best fragment by scoring method to default model
                pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILDS] = {}
                pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILD_SCORES] = {}
                for dtag in datasets:
                    dataset_autobuild_results: AutobuildResultsInterface = {
                        event_id: autobuild_result
                        for event_id, autobuild_result
                        in autobuild_results.items()
                        if dtag == event_id.dtag
                    }

                    if len(dataset_autobuild_results) == 0:
                        # print("\tNo autobuilds for this dataset!")
                        continue

                    all_scores = {}
                    for event_id, autobuild_result in dataset_autobuild_results.items():
                        for path, score in autobuild_result.scores.items():
                            all_scores[path] = score

                    if len(all_scores) == 0:
                        # print(f"\tNo autobuilds for this dataset!")
                        continue

                    # Select fragment build
                    selected_fragement_path = max(
                        all_scores,
                        key=lambda _path: all_scores[_path],
                    )

                    pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILDS][str(dtag)] = str(selected_fragement_path)
                    pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILD_SCORES][str(dtag)] = float(
                        all_scores[selected_fragement_path])

                    # Copy to pandda models
                    model_path = str(pandda_fs_model.processed_datasets[dtag].input_pdb)
                    pandda_model_path = pandda_fs_model.processed_datasets[
                                            dtag].dataset_models.path / constants.PANDDA_EVENT_MODEL.format(str(dtag))
                    merged_structure = merge_ligand_into_structure_from_paths(model_path, selected_fragement_path)
                    save_pdb_file(merged_structure, pandda_model_path)

            update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

        ###################################################################
        # # Classify Events
        ###################################################################
        console.start_classification()

        # If autobuild results are available, use them
        if get_event_class.tag == "autobuild":
            event_classifications: EventClassificationsInterface = {
                event_id: get_event_class(
                    event,
                    autobuild_results[event_id],
                )
                for event_id, event
                in all_events.items()
            }
        elif get_event_class.tag == "trivial":
            event_classifications: EventClassificationsInterface = {
                event_id: get_event_class(event)
                for event_id, event
                in all_events.items()
            }
        else:
            raise Exception("No event classifier specified!")

        console.summarise_event_classifications(event_classifications)

        update_log(
            pandda_log,
            pandda_args.out_dir / constants.PANDDA_LOG_FILE,
        )

        ###################################################################
        # # Rank Events
        ###################################################################
        console.start_ranking()

        # Rank the events to determine the order the are displated in
        with STDOUTManager('Ranking events...', f'\tDone!'):
            if pandda_args.rank_method == "size":
                all_events_ranked = rank_events_size(all_events, grid)
            elif pandda_args.rank_method == "size_delta":
                raise NotImplementedError()
                # all_events_ranked = rank_events_size_delta()
            elif pandda_args.rank_method == "cnn":
                raise NotImplementedError()
                # all_events_ranked = rank_events_cnn()

            elif pandda_args.rank_method == "autobuild":
                if not pandda_args.autobuild:
                    raise Exception("Cannot rank on autobuilds if autobuild is not set!")
                else:
                    all_events_ranked: EventRankingInterface = rank_events_autobuild(
                        all_events,
                        autobuild_results,
                        datasets,
                        pandda_fs_model,
                    )
            else:
                raise Exception(f"Ranking method: {pandda_args.rank_method} is unknown!")

            update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

        # console.summarise_event_ranking(event_classifications)

        ###################################################################
        # # Assign Sites
        ###################################################################
        console.start_assign_sites()

        # Get the events and assign sites to them
        with STDOUTManager('Assigning sites to each event', f'\tDone!'):
            sites: SitesInterface = get_sites(
                all_events_ranked,
                grid,
            )
            all_events_events = Events.from_sites(all_events_ranked, sites, )

        console.summarise_sites(sites)

        ###################################################################
        # # Output pandda summary information
        ###################################################################
        console.start_run_summary()

        # Output a csv of the events
        with STDOUTManager('Building and outputting event table...', f'\tDone!'):
            event_table: EventTableInterface = EventTable.from_events(all_events_events)
            event_table.save(pandda_fs_model.analyses.pandda_analyse_events_file)

        # Output site table
        with STDOUTManager('Building and outputting site table...', f'\tDone!'):
            site_table: SiteTableInterface = SiteTable.from_events(all_events_events,
                                                                   pandda_args.max_site_distance_cutoff)
            site_table.save(pandda_fs_model.analyses.pandda_analyse_sites_file)

        time_finish = time.time()
        pandda_log[constants.LOG_TIME] = time_finish - time_start

        # Output json log
        with STDOUTManager('Saving json log with detailed information on run...', f'\tDone!'):
            if pandda_args.debug:
                printer.pprint(pandda_log)
            save_json_log(
                pandda_log,
                pandda_args.out_dir / constants.PANDDA_LOG_FILE,
            )

        print(f"PanDDA ran in: {time_finish - time_start}")

    ###################################################################
    # # Handle Exceptions
    ###################################################################
    # If an exception has occured, print relevant information to the console and save the log
    except Exception as e:
        console.print_exception(e, pandda_args.debug)
        console.save(pandda_fs_model.console_log_file)

        pandda_log[constants.LOG_TRACE] = traceback.format_exc()
        pandda_log[constants.LOG_EXCEPTION] = str(e)

        print(f"Saving PanDDA log to: {pandda_args.out_dir / constants.PANDDA_LOG_FILE}")

        save_json_log(
            pandda_log,
            pandda_args.out_dir / constants.PANDDA_LOG_FILE,
        )


if __name__ == '__main__':
    with STDOUTManager('Parsing command line args', '\tParsed command line arguments!'):
        args = PanDDAArgs.from_command_line()
        print(args)
        print(args.only_datasets)
        console.summarise_arguments(args)

    process_pandda(args)
