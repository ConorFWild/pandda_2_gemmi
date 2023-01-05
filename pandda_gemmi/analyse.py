# Base python
import os
import traceback
import time
import pprint
from functools import partial
import json
import pickle

# Scientific python libraries
import joblib

# import ray

## Custom Imports
from pandda_gemmi import constants
from pandda_gemmi.analyse_lib import (
    get_fs_model,
    get_datasets,
    get_structure_factors,
    get_reference,
    get_datasets_smoother,
get_alignments,
    get_shell_results,
    get_autobuild_results,
get_event_classifications,
    get_event_ranking,
    handle_exception
)
from pandda_gemmi.common import Partial, update_log
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.dataset.dataset import GetReferenceDataset
from pandda_gemmi.edalignment.grid import GetGrid
from pandda_gemmi.smiles import GetDatasetSmiles
from pandda_gemmi.pandda_logging import STDOUTManager, log_arguments, PanDDAConsole
from pandda_gemmi.dependencies import check_dependencies
from pandda_gemmi.dataset import (
    StructureFactors,
    SmoothBFactors,
    DatasetsStatistics,
    drop_columns,
    GetDatasets,
    GetReferenceDataset,
)
from pandda_gemmi.edalignment import (GetGrid, GetAlignments,
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
from pandda_gemmi.shells import get_shells_multiple_models
from pandda_gemmi.logs import (
    save_json_log,
)

from pandda_gemmi.pandda_functions import (
    get_dask_client,
    process_global_serial,
    process_global_dask,
    get_shells,
    get_common_structure_factors,
)
from pandda_gemmi.event import GetEventScoreInbuilt, add_sites_to_events, GetEventScoreSize

from pandda_gemmi.autobuild import (
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
    GetAutobuildResultRhofit,
    GetAutobuildResultInbuilt
)
from pandda_gemmi.tables import (
    GetEventTable,
    GetSiteTable,
    SaveEvents
)
from pandda_gemmi.fs import GetPanDDAFSModel, GetShellDirs
from pandda_gemmi.processing import (
    process_shell,
    process_shell_multiple_models,
    analyse_model,
    ProcessLocalRay,
    ProcessLocalSerial,
    ProcessLocalSpawn,
    ProcessLocalThreading,
    DaskDistributedProcessor,
    DistributedProcessor
)

from pandda_gemmi import event_classification
from pandda_gemmi.sites import GetSites
from event_rescoring import RescoreEventsAutobuildScore, RescoreEventsEventScore, RescoreEventsAutobuildRSCC, \
    RescoreEventsSize
from pandda_gemmi.analyse_interface import *

joblib.externals.loky.set_loky_pickler('pickle')
printer = pprint.PrettyPrinter()
console = PanDDAConsole()


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


def get_process_global(pandda_args, distributed_tmp, debug=Debug.DEFAULT):
    if pandda_args.global_processing == "serial":
        process_global = process_global_serial
    elif pandda_args.global_processing == "distributed":
        # client = get_dask_client(
        #     scheduler=pandda_args.distributed_scheduler,
        #     num_workers=pandda_args.distributed_num_workers,
        #     queue=pandda_args.distributed_queue,
        #     project=pandda_args.distributed_project,
        #     cores_per_worker=pandda_args.local_cpus,
        #     distributed_mem_per_core=pandda_args.distributed_mem_per_core,
        #     resource_spec=pandda_args.distributed_resource_spec,
        #     job_extra=pandda_args.distributed_job_extra,
        #     walltime=pandda_args.distributed_walltime,
        #     watcher=pandda_args.distributed_watcher,
        # )
        # process_global = partial(
        #     process_global_dask,
        #     client=client,
        #     tmp_dir=distributed_tmp
        # )
        # process_global = DaskDistributedProcessor(
        #     scheduler=pandda_args.distributed_scheduler,
        #     num_workers=pandda_args.distributed_num_workers,
        #     queue=pandda_args.distributed_queue,
        #     project=pandda_args.distributed_project,
        #     cores_per_worker=pandda_args.local_cpus,
        #     distributed_mem_per_core=pandda_args.distributed_mem_per_core,
        #     resource_spec=pandda_args.distributed_resource_spec,
        #     job_extra=pandda_args.distributed_job_extra,
        #     walltime=pandda_args.distributed_walltime,
        #     watcher=pandda_args.distributed_watcher,
        # )
        process_global = DistributedProcessor(distributed_tmp,
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
                                              debug=debug,
                                              )

    else:
        raise Exception(f"Could not find an implementation of --global_processing: {pandda_args.global_processing}")

    return process_global


def get_process_local(pandda_args):
    if pandda_args.local_processing == "serial":
        process_local = ProcessLocalSerial()

    elif pandda_args.local_processing == "multiprocessing_spawn":
        process_local = ProcessLocalSpawn(pandda_args.local_cpus)

    elif pandda_args.local_processing == "threading":
        process_local = ProcessLocalThreading(pandda_args.local_cpus)

    # elif pandda_args.local_processing == "joblib":
    #     # process_local = partial(process_local_joblib, n_jobs=pandda_args.local_cpus, verbose=50, max_nbytes=None)
    #     process_local = ProcessLocalJoblib(pandda_args.local_cpus)

    # elif pandda_args.local_processing == "multiprocessing_forkserver":
    #     mp.set_start_method("forkserver")
    #     process_local = partial(process_local_multiprocessing, n_jobs=pandda_args.local_cpus, method="forkserver")
    #     # process_local_load = partial(process_local_joblib, int(joblib.cpu_count() * 3), "threads")

    # elif pandda_args.local_processing == "multiprocessing_spawn":
    #     mp.set_start_method("spawn")
    #     process_local = partial(process_local_multiprocessing, n_jobs=pandda_args.local_cpus, method="spawn")
    #     # process_local_load = partial(process_local_joblib, int(joblib.cpu_count() * 3), "threads")
    # elif pandda_args.local_processing == "dask":
    #     client = Client(n_workers=pandda_args.local_cpus)
    #     process_local = partial(
    #         process_local_dask,
    #         client=client
    #     )

    elif pandda_args.local_processing == "ray":
        process_local = ProcessLocalRay(pandda_args.local_cpus)

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
        # print("filter models")
        filters["dissimilar_models"] = FilterDissimilarModels(pandda_args.max_rmsd_to_reference)

    if "large_gaps" in filter_keys:
        # print("filter gaps")
        filters["large_gaps"] = FilterIncompleteModels()

    if "dissimilar_spacegroups" in filter_keys:
        # print("filter sg")
        filters["dissimilar_spacegroups"] = FilterDifferentSpacegroups()

    return FiltersReferenceCompatibility(filters, datasets_validator)


def get_score_events_func(pandda_args: PanDDAArgs) -> GetEventScoreInterface:
    if pandda_args.event_score == "inbuilt":
        return GetEventScoreInbuilt()
    elif pandda_args.event_score == "size":
        return GetEventScoreSize()


def process_pandda(pandda_args: PanDDAArgs, ):
    ###################################################################
    # # Configuration
    ###################################################################
    time_start = time.time()

    # Process args
    # distributed_tmp = Path(pandda_args.distributed_tmp)

    # CHeck dependencies
    # with STDOUTManager('Checking dependencies...', '\tAll dependencies validated!'):
    console.start_dependancy_check()
    failed_dependency_list = check_dependencies(pandda_args)
    if len(failed_dependency_list) != 0:
        console.print_failed_dependencies(failed_dependency_list)
        raise Exception("Missing Dependencies!")
    else:
        console.print_successful_dependency_check()

    # Initialise log
    # with STDOUTManager('Initialising log...', '\tPanDDA log initialised!'):
    console.start_log()
    log_path = pandda_args.out_dir / constants.PANDDA_LOG_FILE
    pandda_log: Dict = {}
    pandda_log[constants.LOG_START] = time.time()
    initial_args = log_arguments(pandda_args, )

    pandda_log[constants.LOG_ARGUMENTS] = initial_args
    console.started_log(log_path)

    # Get local processor
    console.start_initialise_multiprocessor()
    process_local: ProcessorInterface = get_process_local(pandda_args)
    console.print_initialized_local_processor(pandda_args)

    # Get reflection smoothing function
    smooth_func: SmoothBFactorsInterface = get_smooth_func(pandda_args)

    # Get XMap loading functions
    load_xmap_func: LoadXMapInterface = get_load_xmap_func(pandda_args)
    load_xmap_flat_func: LoadXMapFlatInterface = get_load_xmap_flat_func(pandda_args)

    # Get the Smile generating function
    get_dataset_smiles: GetDatasetSmilesInterface = GetDatasetSmiles()

    # Get the filtering functions
    datasets_validator: DatasetsValidatorInterface = DatasetsValidator(pandda_args.min_characterisation_datasets)
    filter_data_quality: FiltersDataQualityInterface = get_filter_data_quality(
        args.data_quality_filters,
        datasets_validator,
        pandda_args
    )
    filter_reference_compatability: FiltersReferenceCompatibilityInterface = get_filter_reference_compatability(
        args.reference_comparability_filters,
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
            autobuild_func: Optional[GetAutobuildResultInterface] = GetAutobuildResultInbuilt()

        else:
            raise Exception(f"Autobuild strategy: {pandda_args.autobuild_strategy} is not valid!")

    # Get the rescoring function
    if pandda_args.rescore_event_method == "size":
        event_rescoring_function = RescoreEventsSize()
    elif pandda_args.rescore_event_method == "autobuild_rscc":
        event_rescoring_function = RescoreEventsAutobuildRSCC()
    elif pandda_args.rescore_event_method == "event_score":
        event_rescoring_function = RescoreEventsEventScore()
    elif pandda_args.rescore_event_method == "autobuild_score":
        event_rescoring_function = RescoreEventsAutobuildScore()
    else:
        raise NotImplementedError(
            f"Event rescoring method \"{pandda_args.rescore_event_method}\" if not a valid event rescoring method.")

    # Get the event classification function
    if pandda_args.autobuild:
        get_event_class: GetEventClassInterface = event_classification.GetEventClassAutobuildScore(0.4, 0.25)
    else:
        get_event_class: GetEventClassInterface = event_classification.GetEventClassTrivial()

    # Get the site determination function
    get_sites: GetSitesInterface = GetSites(pandda_args.max_site_distance_cutoff)

    ###################################################################
    # # Run the PanDDA
    ###################################################################
    try:

        ###################################################################
        # # Get fs model
        ###################################################################
        pandda_fs_model = get_fs_model(pandda_args, console, pandda_log, process_local, get_dataset_smiles, )

        ###################################################################
        # # Load datasets
        ###################################################################
        # Get datasets
        datasets_initial, datasets_statistics = get_datasets(pandda_args, console, pandda_fs_model)

        ###################################################################
        # # Get global processor
        ###################################################################
        console.start_initialise_shell_processor()
        process_global: ProcessorInterface = get_process_global(pandda_args, pandda_fs_model.tmp_dir, pandda_args.debug)
        console.print_initialized_global_processor(pandda_args)

        ###################################################################
        # # If structure factors not given, check if any common ones are available
        ###################################################################
        structure_factors = get_structure_factors(pandda_args, console, get_common_structure_factors, datasets_initial)

        ###################################################################
        # # Data Quality filters
        ###################################################################
        console.start_data_quality_filters()

        datasets_for_filtering: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                                     datasets_initial.items()}

        datasets_quality_filtered: DatasetsInterface = filter_data_quality(datasets_for_filtering, structure_factors)
        console.summarise_filtered_datasets(
            filter_data_quality.filtered_dtags
        )

        ###################################################################
        # # Truncate columns
        ###################################################################
        datasets_wilson: DatasetsInterface = drop_columns(datasets_quality_filtered, structure_factors)

        ###################################################################
        # # Reference Selection
        ###################################################################
        reference = get_reference(pandda_args, console, pandda_log, pandda_fs_model, datasets_wilson,
                                  datasets_statistics)

        ###################################################################
        # # B Factor smoothing
        ###################################################################
        datasets_smoother = get_datasets_smoother(console, smooth_func, process_local, datasets_wilson, reference,
                                                  structure_factors, pandda_log)

        ###################################################################
        # # Reference compatability filters
        ###################################################################
        console.start_reference_comparability_filters()

        datasets_reference: DatasetsInterface = filter_reference_compatability(datasets_smoother, reference)
        datasets: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                       datasets_reference.items()}
        console.summarise_filtered_datasets(
            filter_reference_compatability.filtered_dtags
        )

        ###################################################################
        # # Getting grid
        ###################################################################
        console.start_get_grid()

        # Grid
        # with STDOUTManager('Getting the analysis grid...', f'\tDone!'):
        grid: GridInterface = GetGrid()(reference,
                                        pandda_args.outer_mask,
                                        pandda_args.inner_mask_symmetry,
                                        # sample_rate=pandda_args.sample_rate,
                                        sample_rate=reference.dataset.reflections.get_resolution() / 0.5,
                                        debug=pandda_args.debug
                                        )

        if pandda_args.debug >= Debug.AVERAGE_MAPS:
            with open(pandda_fs_model.pandda_dir / "grid.pickle", "wb") as f:
                pickle.dump(grid, f)

            grid.partitioning.save_maps(
                pandda_fs_model.pandda_dir
            )

        console.summarise_get_grid(grid)

        ###################################################################
        # # Getting alignments
        ###################################################################
        alignments: AlignmentsInterface = get_alignments(pandda_args, console, pandda_log, pandda_fs_model, datasets, reference)

        ###################################################################
        # # Assign comparison datasets
        ###################################################################
        console.start_get_comparators()

        # with STDOUTManager('Deciding on the datasets to characterise the groundstate for each dataset to analyse...',
        #                    f'\tDone!'):
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

        console.summarise_get_comarators(comparators)

        ###################################################################
        # # Get the shells
        ###################################################################
        console.start_get_shells()

        # Partition the Analysis into shells in which all datasets are being processed at a similar resolution for the
        # sake of computational efficiency
        shells: ShellsInterface = get_shells_multiple_models(
            datasets,
            comparators,
            pandda_args.min_characterisation_datasets,
            pandda_args.max_shell_datasets,
            pandda_args.high_res_increment,
            pandda_args.only_datasets,
            debug=pandda_args.debug,
        )
        if pandda_args.debug >= Debug.PRINT_SUMMARIES:
            print('Got shells that support multiple models')
            for shell_res, shell in shells.items():
                print(f'\tShell res: {shell.res}: {shell.test_dtags[:3]}')
                for cluster_num, dtags in shell.train_dtags.items():
                    print(f'\t\t{cluster_num}: {dtags[:5]}')

        pandda_fs_model.shell_dirs = GetShellDirs()(pandda_fs_model.pandda_dir, shells)
        pandda_fs_model.shell_dirs.build()

        if pandda_args.debug >= Debug.PRINT_NUMERICS:
            printer.pprint(shells)

        console.summarise_get_shells(shells)

        ###################################################################
        # # Process shells
        ###################################################################
        shell_results, all_events, event_scores = get_shell_results(pandda_args, console, process_global, process_local,
                                                                    pandda_fs_model, process_shell_multiple_models,
                                                                    load_xmap_func, analyse_model_func,
                                                                    score_events_func, shells, structure_factors,
                                                                    datasets, reference, alignments, grid, pandda_log, )

        ###################################################################
        # # Autobuilding
        ###################################################################
        # Autobuild the results if set to
        autobuild_results = get_autobuild_results(pandda_args, console, process_local, process_global, autobuild_func,
                                                  pandda_fs_model, datasets, all_events, shell_results, pandda_log,
                                                  event_scores, )

        ###################################################################
        # # Rescore Events
        ###################################################################
        console.start_rescoring(pandda_args.rescore_event_method)

        event_scores = {
            event_id: event_score
            for event_id, event_score
            in zip(
                all_events,
                process_local(
                    [
                        Partial(
                            event_rescoring_function).paramaterise(
                            event_id,
                            event,
                            pandda_fs_model,
                            datasets[event_id.dtag],
                            event_scores[event_id],
                            autobuild_results[event_id],
                            grid,
                            debug=pandda_args.debug,
                        )
                        for event_id, event
                        in all_events.items()
                    ],
                )
            )
        }

        console.summarise_rescoring(event_scores)

        ###################################################################
        # # Classify Events
        ###################################################################
        event_classifications = get_event_classifications(pandda_args, console, pandda_log, get_event_class, all_events, autobuild_results)

        ###################################################################
        # # Rank Events
        ###################################################################
        event_ranking = get_event_ranking(pandda_args, console, pandda_fs_model, datasets, grid, all_events,
                                          event_scores,
                                          autobuild_results, pandda_log)

        # console.summarise_event_ranking(event_classifications)

        ###################################################################
        # # Assign Sites
        ###################################################################
        console.start_assign_sites()

        # Get the events and assign sites to them
        # with STDOUTManager('Assigning sites to each event', f'\tDone!'):
        sites: SitesInterface = get_sites(
            all_events,
            grid,
        )
        all_events_sites: EventsInterface = add_sites_to_events(all_events, sites, )

        console.summarise_sites(sites)

        ###################################################################
        # # Output pandda summary information
        ###################################################################
        console.start_run_summary()

        # Save the events to json
        SaveEvents()(
            all_events,
            sites,
            pandda_fs_model.events_json_file
        )

        # Output a csv of the events
        # with STDOUTManager('Building and outputting event table...', f'\tDone!'):
        # event_table: EventTableInterface = EventTable.from_events(all_events_sites)
        console.start_event_table_output()
        event_table: EventTableInterface = GetEventTable()(
            all_events,
            sites,
            event_ranking,
        )
        event_table.save(pandda_fs_model.analyses.pandda_analyse_events_file)
        console.summarise_event_table_output(pandda_fs_model.analyses.pandda_analyse_events_file)

        # Output site table
        # with STDOUTManager('Building and outputting site table...', f'\tDone!'):
        console.start_site_table_output()

        # site_table: SiteTableInterface = SiteTable.from_events(all_events_sites,
        #                                                        pandda_args.max_site_distance_cutoff)
        site_table: SiteTableInterface = GetSiteTable()(all_events,
                                                        sites,
                                                        pandda_args.max_site_distance_cutoff)
        site_table.save(pandda_fs_model.analyses.pandda_analyse_sites_file)

        console.summarise_site_table_output(pandda_fs_model.analyses.pandda_analyse_sites_file)

        time_finish = time.time()
        pandda_log[constants.LOG_TIME] = time_finish - time_start

        # Output json log
        console.start_log_save()
        # with STDOUTManager('Saving json log with detailed information on run...', f'\tDone!'):
        if pandda_args.debug >= Debug.PRINT_SUMMARIES:
            printer.pprint(pandda_log)
        save_json_log(
            pandda_log,
            pandda_args.out_dir / constants.PANDDA_LOG_FILE,
        )
        console.summarise_log_save(pandda_args.out_dir / constants.PANDDA_LOG_FILE)

        # print(f"PanDDA ran in: {time_finish - time_start}")
        console.summarise_run(time_finish - time_start)

    ###################################################################
    # # Handle Exceptions
    ###################################################################
    # If an exception has occured, print relevant information to the console and save the log
    except Exception as e:
        handle_exception(pandda_args, console, e, pandda_log)





if __name__ == '__main__':
    # with STDOUTManager('Parsing command line args', '\tParsed command line arguments!'):
    console.start_pandda()

    # Parse Command Line Arguments
    console.start_parse_command_line_args()
    args = PanDDAArgs.from_command_line()
    console.summarise_arguments(args)

    # Process the PanDDA
    process_pandda(args)
