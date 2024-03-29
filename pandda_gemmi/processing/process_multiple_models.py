from __future__ import annotations

# Base python
import dataclasses
import time
import pprint
from functools import partial
import os
import json
from typing import Set
import pickle

from pandda_gemmi.processing.process_local import ProcessLocalSerial
from pandda_gemmi.pandda_logging import STDOUTManager, log_arguments, PanDDAConsole
console = PanDDAConsole()

printer = pprint.PrettyPrinter()

# Scientific python libraries
# import ray

## Custom Imports
from pandda_gemmi.logs import (
    summarise_array,
)

from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.pandda_functions import (
    process_local_serial,
    truncate,
    save_native_frame_zmap,
    save_reference_frame_zmap,
)
from pandda_gemmi.python_types import *
from pandda_gemmi.common import Dtag, EventID, Partial
from pandda_gemmi.fs import PanDDAFSModel, MeanMapFile, StdMapFile
from pandda_gemmi.dataset import (StructureFactors, Dataset, Datasets,
                                  Resolution, )
from pandda_gemmi.shells import Shell, ShellMultipleModels
from pandda_gemmi.edalignment import Partitioning, Xmap, XmapArray, Grid, from_unaligned_dataset_c, GetMapStatistics
from pandda_gemmi.model import Zmap, Model, Zmaps
from pandda_gemmi.event import (
    Event, Clusterings, Clustering, Events, get_event_mask_indicies,
    save_event_map,
)
from pandda_gemmi.density_clustering import (
    GetEDClustering, FilterEDClusteringsSize,
    FilterEDClusteringsPeak,
    MergeEDClusterings,
)


@dataclasses.dataclass()
class DatasetResult(DatasetResultInterface):
    dtag: DtagInterface
    events: MutableMapping[EventIDInterface, EventInterface]
    event_scores: EventScoringResultsInterface
    log: Dict


@dataclasses.dataclass()
class ShellResult(ShellResultInterface):
    shell: ShellInterface
    dataset_results: DatasetResultsInterface
    log: Dict


# TODO: Remove
# def blobfind_event_map_and_report_and_output(
#         test_dtag,
#         model_number,
#         dataset,
#         xmaps,
#         zmap,
#         selected_model_clusterings,
#         model,
#         dataset_xmaps,
#         grid,
#         alignments,
#         max_site_distance_cutoff,
#         min_bdc, max_bdc,
#         reference,
#         contour_level,
#         cluster_cutoff_distance_multiplier,
#         pandda_fs_model
# ):
#     # Get the events and their BDCs
#     events: Events = Events.from_clusters(
#         selected_model_clusterings,
#         model,
#         dataset_xmaps,
#         grid,
#         alignments[test_dtag],
#         max_site_distance_cutoff,
#         min_bdc, max_bdc,
#         None,
#     )
#
#     # Calculate the event maps
#     reference_xmap_grid = xmaps[test_dtag].xmap
#     reference_xmap_grid_array = np.array(reference_xmap_grid, copy=True)
#
#     for event_id, event in events.events.items():
#
#         event_map_reference_grid = gemmi.FloatGrid(*[reference_xmap_grid.nu,
#                                                      reference_xmap_grid.nv,
#                                                      reference_xmap_grid.nw,
#                                                      ]
#                                                    )
#         event_map_reference_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")  # xmap.xmap.spacegroup
#         event_map_reference_grid.set_unit_cell(reference_xmap_grid.unit_cell)
#
#         event_map_reference_grid_array = np.array(event_map_reference_grid,
#                                                   copy=False,
#                                                   )
#
#         mean_array = model.mean
#         event_map_reference_grid_array[:, :, :] = (reference_xmap_grid_array - (event.bdc.bdc * mean_array)) / (
#                 1 - event.bdc.bdc)
#
#         # Mask
#
#         # # Blobfind
#         # clustering = Clustering.from_event_map(
#         #     event_map_reference_grid_array,
#         #     zmap,
#         #     reference,
#         #     grid,
#         #     contour_level,
#         #     cluster_cutoff_distance_multiplier,
#         # )
#
#         scores = score_clusters(
#             {(0, 0): event.cluster},
#             Zmap(event_map_reference_grid),
#             dataset,
#
#         )
#
#         # Ouptut
#         # for cluster_id, cluster in clustering.clustering.items():
#         #     string = f"\t\tModel {model_number} Event {event_id.event_idx.event_idx} Cluster {cluster_id} size: {cluster.values.size} reference frame coords {cluster.centroid}"
#         #     print(string)
#         for score_id, score in scores:
#             string = f"\t\tModel {model_number} Event {event_id.event_idx.event_idx} Score {score}"
#
#         # Save event map
#         filename = f'event_{model_number}_{event_id.event_idx.event_idx}_ref.ccp4'
#         save_reference_frame_zmap(
#             pandda_fs_model.processed_datasets.processed_datasets[test_dtag].z_map_file.path.parent / filename,
#             Zmap(event_map_reference_grid)
#         )
#
#         # Save z map
#         filename = f'z_{model_number}_{event_id.event_idx.event_idx}_ref.ccp4'
#         save_reference_frame_zmap(
#             pandda_fs_model.processed_datasets.processed_datasets[test_dtag].z_map_file.path.parent / filename,
#             zmap
#         )
#
#         # Save reference model
#         filename = f'ref.pdb'
#         reference_structure = reference.dataset.structure.structure
#         reference_structure.write_minimal_pdb(
#             str(pandda_fs_model.processed_datasets.processed_datasets[test_dtag].z_map_file.path.parent / filename)
#         )


def update_log(shell_log, shell_log_path):
    if shell_log_path.exists():
        os.remove(shell_log_path)

    with open(shell_log_path, "w") as f:
        json.dump(shell_log, f, indent=4)


# def get_event_clusters(clusterings_list: Dict[int, Clusterings]):
#
#     cluster_ids = []
#     event_coords = []
#     for model_id, clusterings in clusterings_list.items():
#         for clustering_id, clustering in clusterings.clusterings.items():
#             for cluster_id, cluster in clustering.clustering.items():
#                 cluster_ids.append((model_id, cluster_id))
#                 event_coords.append(cluster.centroid)
#
#     centroids_array = np.array(event_coords)
#
#
#
#
#     ...

# def select_model(model_results: Dict[int, Dict], inner_mask, processed_dataset, debug=False):
#     log = {}
#
#     biggest_clusters = {}
#     for model_number, model_result in model_results.items():
#         cluster_sizes = {}
#         for clustering_id, clustering in model_result['clusterings_large'].clusterings.items():
#             for cluster_id, cluster in clustering.clustering.items():
#                 cluster_sizes[(int(model_number), int(cluster_id))] = int(cluster.values.size)
#
#         if len(cluster_sizes) == 0:
#             continue
#         else:
#             biggest_clusters[model_number] = max(cluster_sizes, key=lambda _x: cluster_sizes[_x])
#
#     zmaps = {}
#     clusters = {}
#     for model_number, model_result in model_results.items():
#         for clustering_id, clustering in model_result['clusterings_large'].clusterings.items():
#             for cluster_id, cluster in clustering.clustering.items():
#                 new_cluster_id = (int(model_number), int(cluster_id),)
#
#                 if new_cluster_id == biggest_clusters[model_number]:
#                     clusters[new_cluster_id] = cluster
#
#                     zmap: Zmap = model_result['zmap']
#                     zmap_array = np.array(zmap.zmap, copy=False)
#
#                     zmap_grid: gemmi.FloatGrid = zmap.grid_from_grid_template(zmap.zmap, zmap_array)
#
#                     zmap_grid_array = np.array(zmap_grid, copy=False)
#
#                     inner_mask_array = np.array(inner_mask, copy=True, dtype=np.int8)
#
#                     zmap_grid_array[np.nonzero(inner_mask_array)] = 0.0
#
#                     zmaps[new_cluster_id] = zmap_grid
#
#     # Score the top clusters
#     results = score_clusters(clusters, zmaps, processed_dataset, debug=debug)
#
#     log = {score_key[0]: score for score_key, score in scores.items()}
#
#     if len(scores) == 0:
#         return 0, log
#
#     else:
#         selected_model_number = max(
#             scores,
#             key=lambda _score: scores[_score],
#         )[0]
#
#     return selected_model_number, log


class ModelSelection(ModelSelectionInterface):
    def __init__(self, selected_model_id: ModelIDInterface, log: Dict) -> None:
        self.selected_model_id = selected_model_id
        self.log = log


def EXPERIMENTAL_select_model(
        model_results: ModelResultsInterface,
        inner_mask: CrystallographicGridInterface,
        processed_dataset: ProcessedDatasetInterface,
        debug: Debug = Debug.DEFAULT,
) -> ModelSelectionInterface:
    log = {}

    model_event_scores: Dict[ModelIDInterface, Dict[EventIDInterface, EventScoringResultInterface]] = {
        model_id: model.event_scores for
        model_id,
        model in model_results.items()
    }

    if debug >= Debug.PRINT_NUMERICS:
        print("Event scores for each model are:")
        print(model_event_scores)



        for model_id, event_scores in model_event_scores.items():
            print(f"Best score for model: {model_id}")
            print(
                [
                    event_scores[score_id].get_selected_structure_score()
                    for score_id
                    in event_scores
                ]
            )
            for event_id, event_score_result in event_scores.items():
                print(f"event log: {event_id.event_idx.event_idx} {event_id.dtag.dtag}")
                print(event_score_result.log())


    # Score the top clusters#
    model_scores = {}
    for model_id, event_scores in model_event_scores.items():
        selected_event_scores = [
            event_scores[event_id].get_selected_structure_score()
            for event_id
            in event_scores
        ]
        if debug >= Debug.PRINT_NUMERICS:
            print(f"\tModel {model_id} all scores: {selected_event_scores}")

        filtered_model_scores = [
                                    selected_event_score
                                    for selected_event_score
                                    in selected_event_scores
                                    if selected_event_score
                                ] + [-0.001, ]
        if debug >= Debug.PRINT_NUMERICS:
            print(f"\tModel {model_id}: filtered scores: {filtered_model_scores}")

        maximum_event_score = max(
            filtered_model_scores
        )
        model_scores[model_id] = maximum_event_score

    # model_scores = {
    #     model_id: max(
    #         [
    #             event_scores[score_id].get_selected_structure_score()
    #             for score_id
    #             in event_scores
    #             if event_scores[score_id].get_selected_structure_score() is not None
    #         ] + [0.0, ])
    #     for model_id, event_scores
    #     in model_event_scores.items()
    # }

    if debug >= Debug.PRINT_SUMMARIES:
        print("Maximum score of any event for each model are are:")
        print(model_scores)

    log['model_scores'] = {
        model_id: float(score)
        for model_id, score
        in model_scores.items()
    }

    if len(model_scores) == 0:
        return ModelSelection(0, log)

    else:
        selected_model_number = max(
            model_scores,
            key=lambda _score: model_scores[_score],
        )  # [0]

    return ModelSelection(selected_model_number, log)

class SelectModel:

    def __call__(self, model_results: ModelResultsInterface,):
        return EXPERIMENTAL_select_model(
            model_results,
        )


def get_models(
        test_dtags: List[DtagInterface],
        comparison_sets: Dict[int, List[DtagInterface]],
        shell_xmaps: XmapsInterface,
        grid: GridInterface,
        process_local: ProcessorInterface,
):
    masked_xmap_array = XmapArray.from_xmaps(
        shell_xmaps,
        grid,
    )

    models = {}
    for comparison_set_id, comparison_set_dtags in comparison_sets.items():
        # comparison_set_dtags =

        # Get the relevant dtags' xmaps
        masked_train_characterisation_xmap_array: XmapArray = masked_xmap_array.from_dtags(
            comparison_set_dtags)
        masked_train_all_xmap_array: XmapArray = masked_xmap_array.from_dtags(
            comparison_set_dtags + [test_dtag for test_dtag in test_dtags])

        mean_array: np.ndarray = Model.mean_from_xmap_array(masked_train_characterisation_xmap_array,
                                                            )  # Size of grid.partitioning.total_mask > 0
        # dataset_log[constants.LOG_DATASET_MEAN] = summarise_array(mean_array)
        # update_log(dataset_log, dataset_log_path)

        sigma_is: Dict[Dtag, float] = Model.sigma_is_from_xmap_array(masked_train_all_xmap_array,
                                                                     mean_array,
                                                                     1.5,
                                                                     )  # size of n
        # dataset_log[constants.LOG_DATASET_SIGMA_I] = {_dtag.dtag: float(sigma_i) for _dtag, sigma_i in sigma_is.items()}
        # update_log(dataset_log, dataset_log_path)

        sigma_s_m: np.ndarray = Model.sigma_sms_from_xmaps(masked_train_characterisation_xmap_array,
                                                           mean_array,
                                                           sigma_is,
                                                           process_local,
                                                           )  # size of total_mask > 0
        # dataset_log[constants.LOG_DATASET_SIGMA_S] = summarise_array(sigma_s_m)
        # update_log(dataset_log, dataset_log_path)

        model: Model = Model.from_mean_is_sms(
            mean_array,
            sigma_is,
            sigma_s_m,
            grid,
        )
        models[comparison_set_id] = model

    return models


class ModelResult(ModelResultInterface):
    def __init__(self,
                 zmap,
                 clusterings,
                 clusterings_large,
                 clusterings_peaked,
                 clusterings_merged,
                 events,
                 event_scores,
                 model_log) -> None:
        self.zmap = zmap
        self.clusterings = clusterings
        self.clusterings_large = clusterings_large
        self.clusterings_peaked = clusterings_peaked
        self.clusterings_merged = clusterings_merged
        self.events = events
        self.event_scores: EventScoringResultsInterface = event_scores
        self.model_log = model_log


def analyse_model(
        model,
        model_number,
        test_dtag,
        dataset_xmap,
        reference,
        grid,
        dataset_processed_dataset,
        dataset_alignment,
        max_site_distance_cutoff,
        min_bdc, max_bdc,
        contour_level,
        cluster_cutoff_distance_multiplier,
        min_blob_volume,
        min_blob_z_peak,
        output_dir,
        score_events_func: GetEventScoreInterface,
        res, rate,
        debug: Debug = Debug.DEFAULT
) -> ModelResultInterface:
    if debug >= Debug.PRINT_SUMMARIES:
        print(f'\tAnalysing model: {model_number}')

    model_log = {}

    # model_log[constants.LOG_DATASET_TRAIN] = [_dtag.dtag for _dtag in shell.train_dtags[model_number]]
    # update_log(dataset_log, dataset_log_path)

    # masked_xmap_array = XmapArray.from_xmaps(
    #     dataset_xmaps,
    #     grid,
    # )

    # masked_train_xmap_array: XmapArray = masked_xmap_array.from_dtags(
    #     [_dtag for _dtag in shell.train_dtags[test_dtag].union({test_dtag, })])

    ###################################################################
    # # Generate the statistical model of the dataset
    ###################################################################
    time_model_analysis_start = time.time()

    # Calculate z maps
    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\tCalculating zmaps")
    time_z_maps_start = time.time()
    zmaps: ZmapsInterface = Zmaps.from_xmaps(
        model=model,
        xmaps={test_dtag: dataset_xmap, },
        model_number=model_number,
        debug=debug,
    )

    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\tCalculated zmaps")

    time_z_maps_finish = time.time()
    model_log[constants.LOG_DATASET_Z_MAPS_TIME] = time_z_maps_finish - time_z_maps_start
    for dtag, zmap in zmaps.items():
        z_map_statistics = GetMapStatistics(
            zmap
        )
        model_log["ZMap statistics"] = {
            "mean": str(z_map_statistics.mean),
            "std": str(z_map_statistics.std),
            ">1.0": str(z_map_statistics.greater_1),
            ">2.0": str(z_map_statistics.greater_2),
            ">3.0": str(z_map_statistics.greater_3),
        }
        if debug >= Debug.PRINT_SUMMARIES:
            print(model_log["ZMap statistics"])

    # update_log(dataset_log, dataset_log_path)

    ###################################################################
    # # Cluster the outlying density
    ###################################################################
    time_cluster_start = time.time()

    # Get the clustered electron desnity outliers

    time_cluster_z_start = time.time()

    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\tClustering")

    clusterings_list: List[EDClusteringInterface] = process_local_serial(
        [
            Partial(GetEDClustering()).paramaterise(
                zmaps[dtag],
                reference=reference,
                grid=grid,
                contour_level=contour_level,
                cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier, )
            for dtag
            in zmaps
        ]
    )
    time_cluster_z_finish = time.time()

    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\tClustering finished")

    # if debug:
    #     model_log['Time to perform primary clustering of z map'] = time_cluster_z_finish - time_cluster_z_start
    #     model_log['time_event_mask'] = {}
    #     for j, clustering in enumerate(clusterings_list):
    #         model_log['time_cluster'] = clustering.time_cluster
    #         model_log['time_np'] = clustering.time_np
    #         model_log['time_event_masking'] = clustering.time_event_masking
    #         model_log['time_get_orth'] = clustering.time_get_orth
    #         model_log['time_fcluster'] = clustering.time_fcluster
    #         for cluster_num, cluster in clustering.clustering.items():
    #             model_log['time_event_mask'][int(cluster_num)] = cluster.time_event_mask

    clusterings: EDClusteringsInterface = {dtag: clustering for dtag, clustering in zip(zmaps, clusterings_list)}

    model_log[constants.LOG_DATASET_INITIAL_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings.values()])
    # update_log(dataset_log, dataset_log_path)
    cluster_sizes = {}
    for dtag, clustering in clusterings.items():
        for cluster_num, cluster in clustering.clustering.items():
            cluster_sizes[int(cluster_num)] = {
                "size": float(cluster.size(grid)),
                "centroid": (float(cluster.centroid[0]), float(cluster.centroid[1]), float(cluster.centroid[2])),
            }
    model_log[constants.LOG_DATASET_CLUSTER_SIZES] = {
        cluster_num: cluster_sizes[cluster_num]
        for j, cluster_num
        in enumerate(sorted(
            cluster_sizes, key=lambda _cluster_num: cluster_sizes[_cluster_num]["size"],
            reverse=True,
        ))
        if j < 10
    }
    # update_log(dataset_log, dataset_log_path)

    # Filter out small clusters
    clusterings_large: EDClusteringsInterface = FilterEDClusteringsSize()(clusterings,
                                                                          grid,
                                                                          min_blob_volume,
                                                                          )
    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\tAfter filtering: large: {}".format(
            {dtag: len(cluster) for dtag, cluster in
             zip(clusterings_large, clusterings_large.values())}))
    model_log[constants.LOG_DATASET_LARGE_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings_large.values()])
    # update_log(dataset_log, dataset_log_path)

    # Filter out weak clusters (low peak z score)
    clusterings_peaked: EDClusteringsInterface = FilterEDClusteringsPeak()(clusterings_large,
                                                                           grid,
                                                                           min_blob_z_peak)
    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\tAfter filtering: peak: {}".format(
            {dtag: len(cluster) for dtag, cluster in
             zip(clusterings_peaked, clusterings_peaked.values())}))
    model_log[constants.LOG_DATASET_PEAKED_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings_peaked.values()])
    # update_log(dataset_log, dataset_log_path)

    # Add the event mask
    for clustering_id, clustering in clusterings_peaked.items():
        for cluster_id, cluster in clustering.clustering.items():
            cluster.event_mask_indicies = get_event_mask_indicies(
                zmaps[test_dtag],
                cluster.cluster_positions_array)

    # Merge the clusters
    clusterings_merged: EDClusteringsInterface = MergeEDClusterings()(clusterings_peaked)
    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\tAfter filtering: merged: {}".format(
            {dtag: len(_cluster) for dtag, _cluster in
             zip(clusterings_merged, clusterings_merged.values())}))
    model_log[constants.LOG_DATASET_MERGED_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings_merged.values()])
    # update_log(dataset_log, dataset_log_path)

    # Log the clustering
    time_cluster_finish = time.time()
    model_log[constants.LOG_DATASET_CLUSTER_TIME] = time_cluster_finish - time_cluster_start
    # update_log(dataset_log, dataset_log_path)

    # TODO: REMOVE: event blob analysis
    # blobfind_event_map_and_report_and_output(
    #     test_dtag,
    #     model_number,
    #     dataset_truncated_datasets[test_dtag],
    #     dataset_xmaps,
    #     zmaps[test_dtag],
    #     clusterings_large,
    #     model,
    #     dataset_xmaps,
    #     grid,
    #     alignments,
    #     max_site_distance_cutoff,
    #     min_bdc, max_bdc,
    #     reference,
    #     contour_level,
    #     cluster_cutoff_distance_multiplier,
    #     pandda_fs_model
    # )

    events: Events = Events.from_clusters(
        clusterings_large,
        model,
        {test_dtag: dataset_xmap, },
        grid,
        dataset_alignment,
        max_site_distance_cutoff,
        min_bdc, max_bdc,
        None,
    )

    if debug >= Debug.PRINT_SUMMARIES:
        print("\t\tScoring events...")

    if debug >= Debug.AVERAGE_MAPS:
        with open(output_dir / "test_dtag.pickle", "wb") as f:
            pickle.dump(test_dtag, f)

        with open(output_dir / "model_number.pickle", "wb") as f:
            pickle.dump(model_number, f)

        with open(output_dir / "dataset_processed_dataset.pickle", "wb") as f:
            pickle.dump(dataset_processed_dataset, f)

        with open(output_dir / "dataset_xmap.pickle", "wb") as f:
            pickle.dump(dataset_xmap, f)

        with open(output_dir / f"events_{model_number}.pickle", "wb") as f:
            pickle.dump(events, f)

        with open(output_dir / f"model_{model_number}.pickle", "wb") as f:
            pickle.dump(model, f)

    if score_events_func.tag == "inbuilt":
        event_scores: EventScoringResultsInterface = score_events_func(
            test_dtag,
            model_number,
            dataset_processed_dataset,
            dataset_xmap,
            zmaps[test_dtag],
            events,
            model,
            grid,
            dataset_alignment,
            max_site_distance_cutoff,
            min_bdc, max_bdc,
            reference,
            res, rate,
            event_map_cut=2.0,
            structure_output_folder=output_dir,
            debug=debug
        )
    # elif score_events_func.tag == "autobuild":
    #     raise NotImplementedError()

    elif score_events_func.tag == "size":
        event_scores: EventScoringResultsInterface = score_events_func(
            events
        )

    else:
        raise Exception("No valid event selection score method!")

    # model_log['score'] = {}
    # model_log['noise'] = {}
    #
    # for event_id, event_scoring_result in event_scores.items():
    #     model_log['score'][int(event_id.event_idx)] = event_scoring_result.get_selected_structure_score()
    # model_log['noise'][int(event_num)] = noises[event_num]

    # event_scores, noises = event_score_autobuild(
    #     test_dtag,
    #     model_number,
    #     dataset_processed_dataset,
    #     dataset_xmap,
    #     events,
    #     model,
    #     grid,
    #     dataset_alignment,
    #     max_site_distance_cutoff,
    #     min_bdc, max_bdc,
    #     reference,
    #     structure_output_folder=output_dir,
    #     debug=debug
    # )

    model_log['score'] = {}
    model_log['noise'] = {}

    for event_id, event_scoring_result in event_scores.items():
        model_log['score'][int(event_id.event_idx)] = event_scoring_result.get_selected_structure_score()
        model_log[int(event_id.event_idx)] = event_scoring_result.log()
        # model_log['noise'][int(event_num)] = noises[event_num]

    time_model_analysis_finish = time.time()

    # model_results = {
    #     'zmap': zmaps[test_dtag],
    #     'clusterings': clusterings,
    #     'clusterings_large': clusterings_large,
    #     'clusterings_peaked': clusterings_peaked,
    #     'clusterings_merged': clusterings_merged,
    #     'events': events,
    #     'event_scores': event_scores,
    #     'log': model_log
    # }
    model_results: ModelResult = ModelResult(
        zmaps[test_dtag],
        clusterings,
        clusterings_large,
        clusterings_peaked,
        clusterings_merged,
        {event_id: event for event_id, event in events.events.items()},
        event_scores,
        model_log
    )

    model_log["Model analysis time"] = time_model_analysis_finish - time_model_analysis_start
    if debug >= Debug.PRINT_SUMMARIES:
        print(f"\t\tModel analysis time: {time_model_analysis_finish - time_model_analysis_start}")

    if debug >= Debug.PRINT_SUMMARIES:
        for event_id, event_score_result in model_results.event_scores.items():
            print(f"event log: {event_id.event_idx.event_idx} {event_id.dtag.dtag}")
            print(event_score_result.log())

    return model_results


def dump_and_load(ob, name):
    print(f"Testing: {name}")

    # time_dump_start = time.time()
    # dumps = pickle.dumps(ob)
    # time_dump_finish = time.time()
    # print(f"\tDump time is: {time_dump_finish - time_dump_start}")
    #
    # time_load_start = time.time()
    # loaded = pickle.loads(dumps)
    # time_load_finish = time.time()
    # print(f"\tLoad time is: {time_load_finish - time_load_start}")

    time_dump_start = time.time()
    with open(f"{name}.pickle", 'wb') as f:
        pickle.dump(ob, f)
    time_dump_finish = time.time()
    print(f"\tDump to disk time is: {time_dump_finish - time_dump_start}")

    time_load_start = time.time()
    with open(f"{name}.pickle", 'rb') as f:
        loaded = pickle.load(f)
    time_load_finish = time.time()
    print(f"\tLoad from disk time is: {time_load_finish - time_load_start}")


def process_dataset_multiple_models(
        test_dtag: DtagInterface,
        models: ModelsInterface,
        shell: ShellInterface,
        dataset_truncated_datasets: DatasetsInterface,
        alignments: AlignmentsInterface,
        dataset_xmaps: XmapsInterface,
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
        score_events_func: GetEventScoreInterface,
        process_local: ProcessorInterface,
        debug: Debug = Debug.DEFAULT,
) -> DatasetResultInterface:
    if debug >= Debug.PRINT_SUMMARIES:
        print(f'\tProcessing dtag: {test_dtag}')
    time_dataset_start = time.time()

    dataset_log_path = pandda_fs_model.processed_datasets.processed_datasets[test_dtag].log_path
    dataset_log = {}
    dataset_log["Model analysis time"] = {}

    ###################################################################
    # # Process the models...
    ###################################################################
    time_model_analysis_start = time.time()

    model_results: ModelResultsInterface = {
        model_number: model_result
        for model_number, model_result
        in zip(
            models,
            process_local(
                [
                    Partial(
                        analyse_model_func).paramaterise(
                        model,
                        model_number,
                        test_dtag=test_dtag,
                        dataset_xmap=dataset_xmaps[test_dtag],
                        reference=reference,
                        grid=grid,
                        dataset_processed_dataset=pandda_fs_model.processed_datasets.processed_datasets[test_dtag],
                        dataset_alignment=alignments[test_dtag],
                        max_site_distance_cutoff=max_site_distance_cutoff,
                        min_bdc=min_bdc, max_bdc=max_bdc,
                        contour_level=contour_level,
                        cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
                        min_blob_volume=min_blob_volume,
                        min_blob_z_peak=min_blob_z_peak,
                        output_dir=pandda_fs_model.processed_datasets.processed_datasets[test_dtag].path,
                        score_events_func=score_events_func,
                        res=shell.res,
                        rate=0.5,
                        debug=debug
                    )
                    for model_number, model
                    in models.items()
                ]
            )
        )
    }

    dataset_log["Model logs"] = {model_number: model_result.model_log for model_number, model_result in
                                 model_results.items()}  #

    time_model_analysis_finish = time.time()

    dataset_log["Time to analyse all models"] = time_model_analysis_finish - time_model_analysis_start

    if debug >= Debug.PRINT_SUMMARIES:
        print(f"\tTime to analyse all models: {time_model_analysis_finish - time_model_analysis_start}")
        for model_number, model_result in model_results.items():
            model_time = dataset_log["Model logs"][model_number]["Model analysis time"]
            print(f"\t\tModel {model_number} processed in {model_time}")

    ###################################################################
    # # Decide which model to use...
    ###################################################################
    if debug >= Debug.PRINT_SUMMARIES:
        print(f"\tSelecting model...")
    model_selection: ModelSelectionInterface = EXPERIMENTAL_select_model(
        model_results,
        grid.partitioning.inner_mask,
        pandda_fs_model.processed_datasets.processed_datasets[test_dtag],
        debug=debug,
    )
    selected_model: ModelInterface = models[model_selection.selected_model_id]
    selected_model_clusterings = model_results[model_selection.selected_model_id].clusterings_merged
    zmap = model_results[model_selection.selected_model_id].zmap
    dataset_log['Selected model'] = int(model_selection.selected_model_id)
    dataset_log['Model selection log'] = model_selection.log

    if debug >= Debug.PRINT_SUMMARIES:
        print(f'\tSelected model is: {model_selection.selected_model_id}')

    ###################################################################
    # # Output the z map
    ###################################################################
    time_output_zmap_start = time.time()

    native_grid = dataset_truncated_datasets[test_dtag].reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        # sample_rate=sample_rate,  # TODO: make this d_min/0.5?
        sample_rate=dataset_truncated_datasets[test_dtag].reflections.get_resolution() / 0.5
    )

    partitioning = Partitioning.from_structure_multiprocess(
        dataset_truncated_datasets[test_dtag].structure,
        native_grid,
        outer_mask,
        inner_mask_symmetry,
    )
    # pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save_reference_frame_zmap(zmap)

    save_native_frame_zmap(
        pandda_fs_model.processed_datasets.processed_datasets[test_dtag].z_map_file.path,
        zmap,
        dataset_truncated_datasets[test_dtag],
        alignments[test_dtag],
        grid,
        structure_factors,
        outer_mask,
        inner_mask_symmetry,
        partitioning,
        sample_rate,
    )

    # TODO: Remove altogether
    if debug >= Debug.DATASET_MAPS:
        for model_number, model_result in model_results.items():
            save_reference_frame_zmap(
                pandda_fs_model.processed_datasets.processed_datasets[
                    test_dtag].z_map_file.path.parent / f'{model_number}_ref.ccp4',
                model_result.zmap
            )
            save_native_frame_zmap(
                pandda_fs_model.processed_datasets.processed_datasets[
                    test_dtag].z_map_file.path.parent / f'{model_number}_native.ccp4',
                model_result.zmap,
                dataset_truncated_datasets[test_dtag],
                alignments[test_dtag],
                grid,
                structure_factors,
                outer_mask,
                inner_mask_symmetry,
                partitioning,
                sample_rate,
            )

    # if statmaps:
    #     mean_map_file = MeanMapFile.from_zmap_file(
    #         pandda_fs_model.processed_datasets.processed_datasets[test_dtag].z_map_file)
    #     mean_map_file.save_native_frame_mean_map(
    #         selected_model,
    #         zmap,
    #         dataset_truncated_datasets[test_dtag],
    #         alignments[test_dtag],
    #         grid,
    #         structure_factors,
    #         outer_mask,
    #         inner_mask_symmetry,
    #         partitioning,
    #         sample_rate,
    #     )

    #     std_map_file = StdMapFile.from_zmap_file(pandda_fs_model.processed_datasets.processed_datasets[
    #                                                  test_dtag].z_map_file)
    #     std_map_file.save_native_frame_std_map(
    #         test_dtag,
    #         selected_model,
    #         zmap,
    #         dataset_truncated_datasets[test_dtag],
    #         alignments[test_dtag],
    #         grid,
    #         structure_factors,
    #         outer_mask,
    #         inner_mask_symmetry,
    #         partitioning,
    #         sample_rate,
    #     )
    time_output_zmap_finish = time.time()
    dataset_log['Time to output z map'] = time_output_zmap_finish - time_output_zmap_start

    ###################################################################
    # # Find the events
    ###################################################################
    time_event_start = time.time()
    # Calculate the shell events
    # events: Events = Events.from_clusters(
    #     selected_model_clusterings,
    #     selected_model,
    #     dataset_xmaps,
    #     grid,
    #     alignments[test_dtag],
    #     max_site_distance_cutoff,
    #     min_bdc, max_bdc,
    #     None,
    # )
    events = model_results[model_selection.selected_model_id].events

    time_event_finish = time.time()
    dataset_log[constants.LOG_DATASET_EVENT_TIME] = time_event_finish - time_event_start
    update_log(dataset_log, dataset_log_path)

    ###################################################################
    # # Generate event maps
    ###################################################################
    time_event_map_start = time.time()

    # Save the event maps!
    # printer.pprint(events)
    Events(events).save_event_maps(
        dataset_truncated_datasets,
        alignments,
        dataset_xmaps,
        selected_model,
        pandda_fs_model,
        grid,
        structure_factors,
        outer_mask,
        inner_mask_symmetry,
        sample_rate,
        native_grid,
        mapper=ProcessLocalSerial(),
    )

    if debug >= Debug.DATASET_MAPS:
        for model_number, model_result in model_results.items():
            for event_id, event in model_result.events.items():
                save_event_map(
                    pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].path / f'{model_number}'
                                                                                                f'_{event_id.event_idx.event_idx}.ccp4',
                    dataset_xmaps[event_id.dtag],
                    models[model_number],
                    event,
                    dataset_truncated_datasets[event_id.dtag],
                    alignments[event_id.dtag],
                    grid,
                    structure_factors,
                    outer_mask,
                    inner_mask_symmetry,
                    partitioning,
                    sample_rate,
                )

    time_event_map_finish = time.time()
    dataset_log[constants.LOG_DATASET_EVENT_MAP_TIME] = time_event_map_finish - time_event_map_start
    update_log(dataset_log, dataset_log_path)

    time_dataset_finish = time.time()
    dataset_log[constants.LOG_DATASET_TIME] = time_dataset_finish - time_dataset_start
    update_log(dataset_log, dataset_log_path)

    return DatasetResult(
        dtag=test_dtag,
        events={event_id: event for event_id, event in events.items()},
        event_scores=model_results[model_selection.selected_model_id].event_scores,
        log=dataset_log,
    )


def save_array_to_map_file(
        array: NDArrayInterface,
        template: CrystallographicGridInterface,
        path: Path
):
    spacing = [template.nu, template.nv, template.nw]
    unit_cell = template.unit_cell
    grid = gemmi.FloatGrid(spacing[0], spacing[1], spacing[2])
    grid.set_unit_cell(unit_cell)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    grid_array = np.array(grid, copy=False)
    grid_array[:, :, :] = array[:, :, :]

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True)
    ccp4.setup()
    ccp4.write_ccp4_map(str(path))


def save_xmap(
        xmap: XmapInterface,
        path: Path
):
    xmap.xmap.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = xmap.xmap
    ccp4.update_ccp4_header(2, True)
    ccp4.setup()
    ccp4.write_ccp4_map(str(path))


def save_raw_xmap(
        dataset: DatasetInterface,
        path: Path,
        structure_factors,
        sample_rate,
):
    unaligned_xmap: gemmi.FloatGrid = dataset.reflections.transform_f_phi_to_map(structure_factors.f,
                                                                                 structure_factors.phi,
                                                                                 sample_rate=sample_rate,
                                                                                 )
    unaligned_xmap.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = unaligned_xmap
    ccp4.update_ccp4_header(2, True)
    ccp4.setup()
    ccp4.write_ccp4_map(str(path))


def process_shell_multiple_models(
        shell: ShellInterface,
        datasets: DatasetsInterface,
        alignments: AlignmentsInterface,
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
        score_events_func: GetEventScoreInterface,
        debug: Debug = Debug.DEFAULT,
):
    if debug >= Debug.DEFAULT:
        console.print_starting_process_shell(shell)

    if memory_availability == "very_low":
        process_local_in_shell: ProcessorInterface = ProcessLocalSerial()
        process_local_in_dataset: ProcessorInterface = ProcessLocalSerial()
        process_local_over_datasets: ProcessorInterface = ProcessLocalSerial()
    elif memory_availability == "low":
        process_local_in_shell: ProcessorInterface = process_local
        process_local_in_dataset: ProcessorInterface = process_local
        process_local_over_datasets: ProcessorInterface = ProcessLocalSerial()
    elif memory_availability == "high":
        process_local_in_shell: ProcessorInterface = process_local
        process_local_in_dataset: ProcessorInterface = ProcessLocalSerial()
        process_local_over_datasets: ProcessorInterface = process_local

    else:
        raise Exception(f"memory_availability: {memory_availability}: does not have defined processors")

    time_shell_start = time.time()
    if pandda_fs_model.shell_dirs:
        shell_log_path = pandda_fs_model.shell_dirs.shell_dirs[shell.res].log_path
    else:
        raise Exception(
            "Attempted to find the log path for the shell, but no shell dir added to pandda_fs_model somehow.")
    shell_log = {}

    # Seperate out test and train datasets
    shell_datasets: DatasetsInterface = {
        dtag: dataset
        for dtag, dataset
        in datasets.items()
        if dtag in shell.all_dtags
    }
    shell_log[constants.LOG_SHELL_DATASETS] = [dtag.dtag for dtag in shell_datasets]
    update_log(shell_log, shell_log_path)

    ###################################################################
    # # Homogonise shell datasets by truncation of resolution
    ###################################################################
    if debug >= Debug.DEFAULT:
        console.print_starting_truncating_shells()

    shell_working_resolution: ResolutionInterface = Resolution(
        max([datasets[dtag].reflections.get_resolution() for dtag in shell.all_dtags]))
    shell_truncated_datasets: DatasetsInterface = truncate(
        shell_datasets,
        resolution=shell_working_resolution,
        structure_factors=structure_factors,
    )
    # TODO: REMOVE?
    # shell_truncated_datasets = shell_datasets
    shell_log["Shell Working Resolution"] = shell_working_resolution.resolution

    if debug >= Debug.DEFAULT:
        console.print_summarise_truncating_shells(shell_truncated_datasets)

    ###################################################################
    # # Generate aligned Xmaps
    ###################################################################
    if debug >= Debug.DEFAULT:
        console.print_starting_loading_xmaps()

    time_xmaps_start = time.time()

    xmaps: XmapsInterface = {
        dtag: xmap
        for dtag, xmap
        in zip(
            shell_truncated_datasets,
            process_local_in_shell(
                [
                    Partial(load_xmap_func).paramaterise(
                        shell_truncated_datasets[key],
                        alignments[key],
                        grid=grid,
                        structure_factors=structure_factors,
                        sample_rate=shell.res / 0.5,
                    )
                    for key
                    in shell_truncated_datasets
                ]
            )
        )
    }

    time_xmaps_finish = time.time()
    shell_log[constants.LOG_SHELL_XMAP_TIME] = time_xmaps_finish - time_xmaps_start
    update_log(shell_log, shell_log_path)

    if debug >= Debug.DATASET_MAPS:
        for dtag, xmap in xmaps.items():
            xmap_array = np.array(xmap.xmap)
            save_array_to_map_file(
                xmap_array,
                grid.grid,
                pandda_fs_model.pandda_dir / f"{shell.res}_{dtag}_ref.ccp4"
            )

            save_raw_xmap(
                shell_truncated_datasets[dtag],
                pandda_fs_model.pandda_dir / f"{shell.res}_{dtag}_mov.ccp4",
                structure_factors,
                sample_rate
            )

            # save_xmap(
            #     xmap,
            #     pandda_fs_model.pandda_dir / f"{shell.res}_{dtag}.ccp4"
            # )

    if debug >= Debug.DEFAULT:
        console.print_summarise_loading_xmaps(xmaps, time_xmaps_finish - time_xmaps_start)

    ###################################################################
    # # Get the models to test
    ###################################################################
    if debug >= Debug.DEFAULT:
        console.print_starting_get_models()
    models: ModelsInterface = get_models(
        shell.test_dtags,
        shell.train_dtags,
        xmaps,
        grid,
        process_local_in_shell,
    )

    if debug >= Debug.PRINT_SUMMARIES:
        for model_key, model in models.items():
            save_array_to_map_file(
                model.mean,
                grid.grid,
                pandda_fs_model.pandda_dir / f"{shell.res}_{model_key}_mean.ccp4"
            )

    if debug >= Debug.DEFAULT:
        console.print_summarise_get_models(models)

    ###################################################################
    # # Process each test dataset
    ###################################################################
    # Now that all the data is loaded, get the comparison set and process each test dtag

    # process_dataset_paramaterized =

    if debug >= Debug.DEFAULT:
        console.print_starting_process_datasets()

    # Process each dataset in the shell
    all_train_dtags_unmerged = [_dtag for l in shell.train_dtags.values() for _dtag in l]
    all_train_dtags = []
    for _dtag in all_train_dtags_unmerged:
        if _dtag not in all_train_dtags:
            all_train_dtags.append(_dtag)

    if debug >= Debug.PRINT_NUMERICS:
        print(f"\tAll train datasets are: {all_train_dtags}")
    # dataset_dtags = {_dtag:  for _dtag in shell.test_dtags for n in shell.train_dtags}
    dataset_dtags = {_dtag: [_dtag] + all_train_dtags for _dtag in shell.test_dtags}
    if debug >= Debug.PRINT_NUMERICS:
        print(f"\tDataset dtags are: {dataset_dtags}")
    results: List[DatasetResultInterface] = process_local_over_datasets(
        [
            Partial(
                process_dataset_multiple_models).paramaterise(
                test_dtag,
                # dataset_truncated_datasets={_dtag: shell_truncated_datasets[_dtag] for _dtag in
                #                             dataset_dtags[test_dtag]},
                dataset_truncated_datasets={test_dtag: shell_truncated_datasets[test_dtag] },
                # dataset_xmaps={_dtag: xmaps[_dtag] for _dtag in dataset_dtags[test_dtag]},
                dataset_xmaps={test_dtag: xmaps[test_dtag], },
                models=models,
                shell=shell,
                alignments=alignments,
                pandda_fs_model=pandda_fs_model,
                reference=reference,
                grid=grid,
                contour_level=contour_level,
                cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
                min_blob_volume=min_blob_volume,
                min_blob_z_peak=min_blob_z_peak,
                structure_factors=structure_factors,
                outer_mask=outer_mask,
                inner_mask_symmetry=inner_mask_symmetry,
                max_site_distance_cutoff=max_site_distance_cutoff,
                min_bdc=min_bdc,
                max_bdc=max_bdc,
                # sample_rate=sample_rate,
                sample_rate=shell.res / 0.5,
                statmaps=statmaps,
                analyse_model_func=analyse_model_func,
                score_events_func=score_events_func,
                process_local=process_local_in_dataset,
                debug=debug,
            )

            for test_dtag
            in shell.test_dtags
        ],
    )



    # Update shell log with dataset results
    shell_log[constants.LOG_SHELL_DATASET_LOGS] = {}
    for result in results:
        if result:
            shell_log[constants.LOG_SHELL_DATASET_LOGS][str(result.dtag)] = result.log

    time_shell_finish = time.time()
    shell_log[constants.LOG_SHELL_TIME] = time_shell_finish - time_shell_start
    update_log(shell_log, shell_log_path)

    shell_result: ShellResultInterface = ShellResult(
        shell=shell,
        dataset_results={dtag: result for dtag, result in zip(shell.test_dtags, results) if result},
        log=shell_log,

    )

    if debug >= Debug.DEFAULT:
        console.print_summarise_process_datasets(shell_result)

    return shell_result
