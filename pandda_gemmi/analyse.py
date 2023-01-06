import time
import pprint

# Scientific python libraries
import joblib

# import ray

## Custom Imports
from pandda_gemmi import constants
from pandda_gemmi.analyse_lib import (
    get_process_local,
    get_smooth_func,
    get_load_xmap_func,
    get_load_xmap_flat_func,
    get_filter_data_quality,
    get_filter_reference_compatability,
    get_comparator_func,
    get_analyse_model_func,
    get_score_events_func,
    get_autobuild_func,
    get_event_rescoring_func,
    get_event_classification_func,
    get_process_global,
    get_fs_model,
    get_datasets,
    get_structure_factors,
    get_reference,
    get_datasets_smoother,
    get_grid,
    get_alignments,
    get_comparators,
    get_shells,
    get_shell_results,
    get_autobuild_results,
    get_rescored_events,
    get_event_classifications,
    get_event_ranking,
    handle_exception
)
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.smiles import GetDatasetSmiles
from pandda_gemmi.pandda_logging import log_arguments, PanDDAConsole
from pandda_gemmi.dependencies import check_dependencies
from pandda_gemmi.dataset import (
    drop_columns,
)
from pandda_gemmi.filters import (
    DatasetsValidator
)
from pandda_gemmi.logs import (
    save_json_log,
)
from pandda_gemmi.pandda_functions import (
    get_common_structure_factors,
)
from pandda_gemmi.event import GetEventScoreInbuilt, add_sites_to_events, GetEventScoreSize
from pandda_gemmi.tables import (
    GetEventTable,
    GetSiteTable,
    SaveEvents
)
from pandda_gemmi.processing import (
    process_shell_multiple_models,
)
from pandda_gemmi.sites import GetSites
from pandda_gemmi.analyse_interface import *

joblib.externals.loky.set_loky_pickler('pickle')
printer = pprint.PrettyPrinter()
console = PanDDAConsole()


def process_pandda(pandda_args: PanDDAArgs, ):
    ###################################################################
    # # Configuration
    ###################################################################
    time_start = time.time()

    # CHeck dependencies
    console.start_dependancy_check()
    failed_dependency_list = check_dependencies(pandda_args)
    if len(failed_dependency_list) != 0:
        console.print_failed_dependencies(failed_dependency_list)
        raise Exception("Missing Dependencies!")
    else:
        console.print_successful_dependency_check()

    # Initialise log
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
    autobuild_func: Optional[GetAutobuildResultInterface] = get_autobuild_func(pandda_args)

    # Get the rescoring function
    event_rescoring_function = get_event_rescoring_func(pandda_args)

    # Get the event classification function
    get_event_class = get_event_classification_func(pandda_args)

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
        datasets_quality_filtered = get_data_quality_filtered_datasets(filter_data_quality, datasets_initial,
                                                                       structure_factors)

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
        datasets = get_reference_compatability_filtered_datasets(filter_reference_compatability, datasets_smoother,
                                                                 reference)

        ###################################################################
        # # Getting grid
        ###################################################################
        grid = get_grid(pandda_args, console, pandda_fs_model, reference)

        ###################################################################
        # # Getting alignments
        ###################################################################
        alignments: AlignmentsInterface = get_alignments(pandda_args, console, pandda_log, pandda_fs_model, datasets,
                                                         reference)

        ###################################################################
        # # Assign comparison datasets
        ###################################################################
        comparators = get_comparators(pandda_args, console, pandda_fs_model, pandda_log, comparators_func, datasets,
                                      structure_factors, alignments, grid)

        ###################################################################
        # # Get the shells
        ###################################################################
        shells = get_shells(pandda_args, console, pandda_fs_model, datasets, comparators)

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
        event_scores = get_rescored_events(pandda_fs_model, pandda_args, console, process_local,
                                           event_rescoring_function,
                                           datasets, grid, all_events, event_scores, autobuild_results)

        ###################################################################
        # # Classify Events
        ###################################################################
        event_classifications = get_event_classifications(pandda_args, console, pandda_log, get_event_class, all_events,
                                                          autobuild_results)

        ###################################################################
        # # Rank Events
        ###################################################################
        event_ranking = get_event_ranking(pandda_args, console, pandda_fs_model, datasets, grid, all_events,
                                          event_scores,
                                          autobuild_results, pandda_log)

        ###################################################################
        # # Assign Sites
        ###################################################################
        sites = get_event_sites(get_sites, grid, all_events)

        ###################################################################
        # # Output pandda summary information
        ###################################################################
        get_event_table(pandda_fs_model, all_events, event_ranking, sites)

        get_site_table(pandda_args, pandda_fs_model, all_events, sites)

        summarize_run(pandda_args, pandda_log, time_start)

    ###################################################################
    # # Handle Exceptions
    ###################################################################
    # If an exception has occured, print relevant information to the console and save the log
    except Exception as e:
        handle_exception(pandda_args, console, e, pandda_log)


def summarize_run(pandda_args, pandda_log, time_start):
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


def get_site_table(pandda_args, pandda_fs_model, all_events, sites):
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


def get_event_table(pandda_fs_model, all_events, event_ranking, sites):
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


def get_event_sites(get_sites, grid, all_events):
    console.start_assign_sites()
    # Get the events and assign sites to them
    # with STDOUTManager('Assigning sites to each event', f'\tDone!'):
    sites: SitesInterface = get_sites(
        all_events,
        grid,
    )
    all_events_sites: EventsInterface = add_sites_to_events(all_events, sites, )
    console.summarise_sites(sites)
    return sites


def get_reference_compatability_filtered_datasets(filter_reference_compatability, datasets_smoother, reference):
    console.start_reference_comparability_filters()
    datasets_reference: DatasetsInterface = filter_reference_compatability(datasets_smoother, reference)
    datasets: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                   datasets_reference.items()}
    console.summarise_filtered_datasets(
        filter_reference_compatability.filtered_dtags
    )
    return datasets


def get_data_quality_filtered_datasets(filter_data_quality, datasets_initial, structure_factors):
    console.start_data_quality_filters()
    datasets_for_filtering: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                                 datasets_initial.items()}
    datasets_quality_filtered: DatasetsInterface = filter_data_quality(datasets_for_filtering, structure_factors)
    console.summarise_filtered_datasets(
        filter_data_quality.filtered_dtags
    )
    return datasets_quality_filtered


if __name__ == '__main__':
    # with STDOUTManager('Parsing command line args', '\tParsed command line arguments!'):
    console.start_pandda()

    # Parse Command Line Arguments
    console.start_parse_command_line_args()
    args = PanDDAArgs.from_command_line()
    console.summarise_arguments(args)

    # Process the PanDDA
    process_pandda(args)
