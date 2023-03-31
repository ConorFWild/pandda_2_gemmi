# Base python


# Scientific python libraries

# import ray

## Custom Imports

from pandda_gemmi.args import PanDDAArgs

from pandda_gemmi.dataset import (
    SmoothBFactors,

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
)
from pandda_gemmi.comparators import (
    GetComparatorsHybrid, GetComparatorsHighResFirst, GetComparatorsHighResRandom, GetComparatorsHighRes,
    GetComparatorsCluster)

# from pandda_gemmi.pandda_functions import (
#     process_global_serial,
#
# )
from pandda_gemmi.event import GetEventScoreInbuilt, GetEventScoreSize, GetEventScoreCNN

from pandda_gemmi.processing import (
    analyse_model,
    analyse_model_wrapper,
    ProcessLocalRay,
    ProcessLocalSerial,
    ProcessLocalSpawn,
    ProcessLocalThreading,
    DistributedProcessor,
    DaskDistributedProcessor
)
from pandda_gemmi.autobuild import (
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
    GetAutobuildResultRhofit,
    GetAutobuildResultInbuilt,
    GetAutobuildResultRhofitWrapper
)
from pandda_gemmi import event_classification

from pandda_gemmi.event_rescoring import RescoreEventsAutobuildScore, RescoreEventsEventScore, \
    RescoreEventsAutobuildRSCC, \
    RescoreEventsSize

from pandda_gemmi.analyse_interface import *


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
        process_global = ProcessLocalSerial()
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
                                              cores_per_worker=pandda_args.distributed_cores_per_worker,
                                              distributed_mem_per_core=pandda_args.distributed_mem_per_core,
                                              resource_spec=pandda_args.distributed_resource_spec,
                                              job_extra=pandda_args.distributed_job_extra,
                                              walltime=pandda_args.distributed_walltime,
                                              watcher=pandda_args.distributed_watcher,
                                              debug=debug,
                                              )

    elif pandda_args.global_processing == "dask_distributed":
        process_global = DaskDistributedProcessor(
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
    # analyse_model_func = analyse_model
    if pandda_args.global_processing == "distributed":
        analyse_model_func = analyse_model_wrapper
    else:
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
    elif pandda_args.event_score == "cnn":
        return GetEventScoreCNN()


def get_event_classification_func(pandda_args):
    if pandda_args.autobuild:
        get_event_class: GetEventClassInterface = event_classification.GetEventClassAutobuildScore(0.4, 0.25)
    else:
        get_event_class: GetEventClassInterface = event_classification.GetEventClassTrivial()
    return get_event_class


def get_event_rescoring_func(pandda_args):
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
    return event_rescoring_function


def get_autobuild_func(pandda_args, ):
    if pandda_args.autobuild:

        # with STDOUTManager('Setting up autobuilding...', '\tSet up autobuilding!'):
        if pandda_args.autobuild_strategy == "rhofit":
            # autobuild_func: Optional[GetAutobuildResultInterface] = GetAutobuildResultRhofit()
            autobuild_func = GetAutobuildResultRhofitWrapper()

        elif pandda_args.autobuild_strategy == "inbuilt":
            autobuild_func: Optional[GetAutobuildResultInterface] = GetAutobuildResultInbuilt()


        else:
            raise Exception(f"Autobuild strategy: {pandda_args.autobuild_strategy} is not valid!")

    else:
        autobuild_func = None

    return autobuild_func


def event_criteria_high_scoring(events: EventsInterface, event_scores: EventScoringResultsInterface,
                                # threshold: float = 0.1,
                                # threshold: float = 0.0,
                                threshold: float = 0.05,
                                ):
    return {event_id: event for event_id, event in events.items() if event_scores[event_id].get_selected_structure_score() > threshold}

def event_criteria_all(events: EventsInterface, event_scores: EventScoresInterface):
    return {event_id: event for event_id, event in events.items()}

def get_event_criteria(pandda_args):
    if pandda_args.event_score == "cnn":
        return event_criteria_high_scoring
    else:
        return event_criteria_all


def autobuild_criteria_high_scoring(events: EventsInterface, event_scores: EventScoresInterface,
                                    threshold: float = 0.8):
    return {event_id: event for event_id, event in events.items() if event_scores[event_id] > threshold}


def autobuild_criteria_all(events: EventsInterface,
                           event_scores: EventScoresInterface,
                           threshold: float = 0.8):
    return {event_id: event for event_id, event in events.items()}


def merge_criteria_highest_scoring_event(
        events: EventsInterface,
        event_scores: EventScoresInterface,
        autobuild_results: AutobuildResultsInterface,
        threshold: float = 0.8):
    highest_scoring_event_id = max(event_scores, key=lambda _event_id: event_scores[_event_id])
    all_scores = {}
    for event_id, autobuild_result in autobuild_results.items():
        if event_id != highest_scoring_event_id:
            continue
        for path, autobuild_score in autobuild_result.scores.items():
            all_scores[path] = autobuild_score

    if len(all_scores) == 0:
        return None

    # Select fragment build
    selected_fragement_path = max(
        all_scores,
        key=lambda _path: all_scores[_path],
    )

    return selected_fragement_path


def merge_criteria_highest_scoring_autobuild(
        events: EventsInterface,
        event_scores: EventScoresInterface,
        autobuild_results: AutobuildResultsInterface,
        threshold: float = 0.8):
    all_scores = {}
    for event_id, autobuild_result in autobuild_results.items():
        for path, autobuild_score in autobuild_result.scores.items():
            all_scores[path] = autobuild_score

    if len(all_scores) == 0:
        return None

    # Select fragment build
    selected_fragement_path = max(
        all_scores,
        key=lambda _path: all_scores[_path],
    )

    return selected_fragement_path


# Get the autobuild criterion
def get_autobuild_criterion(pandda_args: PanDDAArgs):
    if pandda_args.event_score == "cnn":
        return autobuild_criteria_high_scoring
    else:
        return autobuild_criteria_all


# Get the autobuild merge criterion
def get_autobuild_merge_criterion(pandda_args: PanDDAArgs):
    if pandda_args.event_score == "cnn":
        return merge_criteria_highest_scoring_event
    else:
        return merge_criteria_highest_scoring_autobuild
