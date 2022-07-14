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
from pandda_gemmi.event import GetEventScoreInbuilt, add_sites_to_events
from pandda_gemmi.ranking import (
    GetEventRankingAutobuild,
    GetEventRankingSize,
    GetEventRankingSizeAutobuild
)
from pandda_gemmi.autobuild import (
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
    GetAutobuildResultRhofit,
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
)

from pandda_gemmi import event_classification
from pandda_gemmi.sites import GetSites
from pandda_gemmi.analyse_interface import *
from pandda_gemmi.pandda import *

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
        process_local = ProcessLocalSerial()

    elif pandda_args.local_processing == "multiprocessing_spawn":
        process_local = ProcessLocalSpawn(pandda_args.local_cpus)

    # elif pandda_args.local_processing == "joblib":
    #     process_local = partial(process_local_joblib, n_jobs=pandda_args.local_cpus, verbose=50, max_nbytes=None)

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
        print("filter models")
        filters["dissimilar_models"] = FilterDissimilarModels(pandda_args.max_rmsd_to_reference)

    if "large_gaps" in filter_keys:
        print("filter gaps")
        filters["large_gaps"] = FilterIncompleteModels()

    if "dissimilar_spacegroups" in filter_keys:
        print("filter sg")
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

    # Get serial processor
    process_serial = ProcessLocalSerial()

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
            "large_gaps",
            # "dissimilar_spacegroups",
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
        get_event_class: GetEventClassInterface = event_classification.GetEventClassAutobuildScore(0.4, 0.25)
    else:
        get_event_class: GetEventClassInterface = event_classification.GetEventClassTrivial()

    # Get the site determination function
    get_sites: GetSitesInterface = GetSites(pandda_args.max_site_distance_cutoff)

    # Function to get the alignments
    get_alignments_func = GetAlignments()

    # Function to get the grid
    get_grid_func = GetGrid()


    ###################################################################
    # # Assemble the PanDDA
    ###################################################################

    # Construct the model processor
    get_zmap = PanDDAGetZmap()
    get_events = PanDDAGetEvents()
    score_events = PanDDAScoreEvents()
    get_model_result = PanDDAGetModelResult()
    process_model = PanDDAProcessModel(
        get_zmap,
        get_events,
        score_events,
        get_model_result,
    )

    # Construct the Dataset processor
    get_model_results = PanDDAGetModelResults()
    select_model = PanDDASelectModel()
    output_maps = PanDDAOutputMaps()
    get_dataset_result = PanDDAGetDatasetResult()
    process_dataset = PanDDAProcessDataset(
        get_model_results,
        select_model,
        output_maps,
        get_dataset_result,
    )

    # Construct the Shell processor
    get_shell_datasets = PanDDAGetShellDatasets()
    homogenise_datasets = PanDDAGetHomogenisedDatasets()
    get_shell_xmaps = PanDDAGetShellXMaps()
    get_models = PanDDAGetModels()
    get_dataset_results = PanDDAGetDatasetResults()
    get_shell_result = PanDDAGetShellResult()
    process_shell = PanDDAProcessShell(
        get_shell_datasets,
        homogenise_datasets,
        get_shell_xmaps,
        get_models,
        get_dataset_results,
        get_shell_result,
        console,
    )

    # Construct the PanDDA
    get_fs_model = PanDDAGetFSModel(
        GetPanDDAFSModel(
            pandda_args.data_dirs,
            pandda_args.out_dir,
            pandda_args.pdb_regex,
            pandda_args.mtz_regex,
            pandda_args.ligand_dir_regex,
            pandda_args.ligand_cif_regex,
            pandda_args.ligand_pdb_regex,
            pandda_args.ligand_smiles_regex,
        ),
        console,
    )
    load_datasets = PanDDALoadDatasets(
        get_datasets_func,
        get_dataset_statistics_func,
        get_common_structure_factors_func,
        get_structure_factors_func,
        pandda_console,
    )
    filter_datasets = PanDDAFilterDatasets(
        filter_data_quality,
        console,
    )
    get_reference = PanDDAGetReference(
        GetReferenceDataset(),
        console,
    )
    filter_reference = PanDDAFilterReference()
    postprocess_datasets = PanDDAPostprocessDatasets()
    get_grid = PanDDAGetGrid(
        get_grid_func, console,
    )
    get_alignments = PanDDAGetAlignments(
        get_alignments_func,
    )
    get_shell_results = PanDDAGetShellResults(
        get_comparators,
        get_shells,
        process_shell,
        console,
    )
    get_autobuilds = PanDDAGetAutobuilds(
        autobuild_func,
        console,
    )
    summarise_run = PanDDASummariseRun(
        get_event_class,
        get_event_ranking,
        get_sites,
        save_events,
        get_event_table,
        get_site_table,
        console,
    )
    pandda = PanDDA(
        get_fs_model,
        load_datasets,
        filter_datasets,
        get_reference,
        filter_reference,
        postprocess_datasets,
        get_grid,
        get_alignments,
        get_shell_results,
        get_autobuilds,
        summarise_run,
    )

    ###################################################################
    # # Run the PanDDA
    ###################################################################
    pandda()


if __name__ == '__main__':
    with STDOUTManager('Parsing command line args', '\tParsed command line arguments!'):
        args = PanDDAArgs.from_command_line()
        print(args)
        print(args.only_datasets)
        console.summarise_arguments(args)

    process_pandda(args)


