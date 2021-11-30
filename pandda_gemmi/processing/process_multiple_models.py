from __future__ import annotations

# Base python
import dataclasses
import time
import pprint
from functools import partial
import os
import json
from typing import Set

printer = pprint.PrettyPrinter()

# Scientific python libraries


## Custom Imports
from pandda_gemmi.logs import (
    summarise_array,
)

from pandda_gemmi import constants
from pandda_gemmi.pandda_functions import (
    process_local_serial,
    truncate,
    save_native_frame_zmap
)
from pandda_gemmi.python_types import *
from pandda_gemmi.common import Dtag, EventID
from pandda_gemmi.fs import PanDDAFSModel, MeanMapFile, StdMapFile
from pandda_gemmi.dataset import (StructureFactors, Dataset, Datasets,
                                  Resolution, )
from pandda_gemmi.shells import Shell, ShellMultipleModels
from pandda_gemmi.edalignment import Partitioning, Xmap, XmapArray, Grid
from pandda_gemmi.model import Zmap, Model, Zmaps
from pandda_gemmi.event import Event, Clusterings, Clustering, Events


@dataclasses.dataclass()
class DatasetResult:
    dtag: Dtag
    events: Dict[EventID, Event]
    log: Dict


@dataclasses.dataclass()
class ShellResult:
    shell: Shell
    dataset_results: Dict[Dtag, DatasetResult]
    log: Dict


def update_log(shell_log, shell_log_path):
    if shell_log_path.exists():
        os.remove(shell_log_path)

    with open(shell_log_path, "w") as f:
        json.dump(shell_log, f)

def select_model(model_results: Dict[int, Dict]):
    model_scores = {}
    for model_number, model_result in model_results.items():
        num_merged_clusters = len(model_result['clusterings_merged'])
        if num_merged_clusters == 0:
            model_score = 0
        elif num_merged_clusters < 6:
            model_score = num_merged_clusters
        elif num_merged_clusters < 12:
            model_score = 0.5
        else:
            model_score = -1

        model_scores[model_number] = model_score
    selected_model = max(
        model_scores,
        key=lambda _number: model_scores[_number],
    )
    return selected_model


def get_models(
        comparison_sets: Dict[int, List[Dtag]],
        shell_xmaps,
        grid: Grid,
        process_local,
):
    masked_xmap_array = XmapArray.from_xmaps(
        shell_xmaps,
        grid,
    )

    models = {}
    for comparison_set_id, comparison_set_dtags in comparison_sets.items():
        # comparison_set_dtags =

        # Get the relevant dtags' xmaps
        masked_train_xmap_array: XmapArray = masked_xmap_array.from_dtags(
            comparison_set_dtags)

        mean_array: np.ndarray = Model.mean_from_xmap_array(masked_train_xmap_array,
                                                            )  # Size of grid.partitioning.total_mask > 0
        # dataset_log[constants.LOG_DATASET_MEAN] = summarise_array(mean_array)
        # update_log(dataset_log, dataset_log_path)

        sigma_is: Dict[Dtag, float] = Model.sigma_is_from_xmap_array(masked_train_xmap_array,
                                                                     mean_array,
                                                                     1.5,
                                                                     )  # size of n
        # dataset_log[constants.LOG_DATASET_SIGMA_I] = {_dtag.dtag: float(sigma_i) for _dtag, sigma_i in sigma_is.items()}
        # update_log(dataset_log, dataset_log_path)

        sigma_s_m: np.ndarray = Model.sigma_sms_from_xmaps(masked_train_xmap_array,
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


def process_dataset_multiple_models(
        test_dtag,
        models,
        shell: ShellMultipleModels,
        dataset_truncated_datasets,
        alignments,
        dataset_xmaps,
        pandda_fs_model: PanDDAFSModel,
        reference,
        grid,
        contour_level,
        cluster_cutoff_distance_multiplier,
        min_blob_volume,
        min_blob_z_peak,
        structure_factors,
        outer_mask,
        inner_mask_symmetry,
        max_site_distance_cutoff,
        min_bdc, max_bdc,
        sample_rate,
        statmaps,
        process_local=process_local_serial,
        debug=False,
):
    if debug:
        print(f'\tProcessing dtag: {test_dtag}')
    time_dataset_start = time.time()

    dataset_log_path = pandda_fs_model.processed_datasets.processed_datasets[test_dtag].log_path
    dataset_log = {}


    model_results = {}
    for model_number, model in models.items():
        if debug:
            print(f'\tAnalysing model: {model_number}')

        dataset_log[constants.LOG_DATASET_TRAIN] = [_dtag.dtag for _dtag in shell.train_dtags[model_number]]
        update_log(dataset_log, dataset_log_path)

        # masked_xmap_array = XmapArray.from_xmaps(
        #     dataset_xmaps,
        #     grid,
        # )

        # masked_train_xmap_array: XmapArray = masked_xmap_array.from_dtags(
        #     [_dtag for _dtag in shell.train_dtags[test_dtag].union({test_dtag, })])

        ###################################################################
        # # Generate the statistical model of the dataset
        ###################################################################
        time_model_start = time.time()

        # Calculate z maps
        time_z_maps_start = time.time()
        zmaps: Dict[Dtag, Zmap] = Zmaps.from_xmaps(
            model=model,
            xmaps={test_dtag: dataset_xmaps[test_dtag], },
        )
        time_z_maps_finish = time.time()
        dataset_log[constants.LOG_DATASET_Z_MAPS_TIME] = time_z_maps_finish - time_z_maps_start
        update_log(dataset_log, dataset_log_path)

        for dtag in zmaps:
            zmap = zmaps[dtag]
            partitioning = Partitioning.from_structure_multiprocess(
                dataset_truncated_datasets[test_dtag].structure,
                grid,
                outer_mask,
                inner_mask_symmetry,
            )
            # pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save_reference_frame_zmap(zmap)

            save_native_frame_zmap(
                pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.path,
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

            if statmaps:
                mean_map_file = MeanMapFile.from_zmap_file(
                    pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file)
                mean_map_file.save_native_frame_mean_map(
                    model,
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

                std_map_file = StdMapFile.from_zmap_file(pandda_fs_model.processed_datasets.processed_datasets[
                                                             dtag].z_map_file)
                std_map_file.save_native_frame_std_map(
                    dtag,
                    model,
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

        ###################################################################
        # # Cluster the outlying density
        ###################################################################
        time_cluster_start = time.time()

        # Get the clustered electron desnity outliers
        cluster_paramaterised = partial(
            Clustering.from_zmap,
            reference=reference,
            grid=grid,
            contour_level=contour_level,
            cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
        )

        clusterings = process_local(
            [
                partial(cluster_paramaterised, zmaps[dtag], )
                for dtag
                in zmaps
            ]
        )
        clusterings = Clusterings({dtag: clustering for dtag, clustering in zip(zmaps, clusterings)})
        # print("\t\tIntially found clusters: {}".format(
        #     {
        #         dtag: (
        #             len(clustering),
        #             max([len(cluster.indexes[0]) for cluster in clustering.clustering.values()] + [0, ]),
        #             max([cluster.size(grid) for cluster in clustering.clustering.values()] + [0, ]),
        #         )
        #         for dtag, clustering in zip(clusterings.clusterings, clusterings.clusterings.values())
        #     }
        # )
        # )
        dataset_log[constants.LOG_DATASET_INITIAL_CLUSTERS_NUM] = sum(
            [len(clustering) for clustering in clusterings.clusterings.values()])
        update_log(dataset_log, dataset_log_path)
        cluster_sizes = {}
        for dtag, clustering in clusterings.clusterings.items():
            for cluster_num, cluster in clustering.clustering.items():
                cluster_sizes[int(cluster_num)] = {
                    "size": float(cluster.size(grid)),
                    "centroid": (float(cluster.centroid[0]), float(cluster.centroid[1]), float(cluster.centroid[2])),
                }
        dataset_log[constants.LOG_DATASET_CLUSTER_SIZES] = {
            cluster_num: cluster_sizes[cluster_num]
            for j, cluster_num
            in enumerate(sorted(
                cluster_sizes, key=lambda _cluster_num: cluster_sizes[_cluster_num]["size"],
                reverse=True,
            ))
            if j < 10
        }
        update_log(dataset_log, dataset_log_path)

        # Filter out small clusters
        clusterings_large: Clusterings = clusterings.filter_size(grid,
                                                                 min_blob_volume,
                                                                 )
        if debug:
            print("\t\tAfter filtering: large: {}".format(
                {dtag: len(cluster) for dtag, cluster in
                 zip(clusterings_large.clusterings, clusterings_large.clusterings.values())}))
        dataset_log[constants.LOG_DATASET_LARGE_CLUSTERS_NUM] = sum(
            [len(clustering) for clustering in clusterings_large.clusterings.values()])
        update_log(dataset_log, dataset_log_path)

        # Filter out weak clusters (low peak z score)
        clusterings_peaked: Clusterings = clusterings_large.filter_peak(grid,
                                                                        min_blob_z_peak)
        if debug:
            print("\t\tAfter filtering: peak: {}".format(
                {dtag: len(cluster) for dtag, cluster in
                 zip(clusterings_peaked.clusterings, clusterings_peaked.clusterings.values())}))
        dataset_log[constants.LOG_DATASET_PEAKED_CLUSTERS_NUM] = sum(
            [len(clustering) for clustering in clusterings_peaked.clusterings.values()])
        update_log(dataset_log, dataset_log_path)

        clusterings_merged = clusterings_peaked.merge_clusters()
        if debug:
            print("\t\tAfter filtering: merged: {}".format(
                {dtag: len(_cluster) for dtag, _cluster in
                 zip(clusterings_merged.clusterings, clusterings_merged.clusterings.values())}))
        dataset_log[constants.LOG_DATASET_MERGED_CLUSTERS_NUM] = sum(
            [len(clustering) for clustering in clusterings_merged.clusterings.values()])
        update_log(dataset_log, dataset_log_path)

        time_cluster_finish = time.time()
        dataset_log[constants.LOG_DATASET_CLUSTER_TIME] = time_cluster_finish - time_cluster_start
        update_log(dataset_log, dataset_log_path)

        model_results[model_number] = {
            'zmap': zmaps[test_dtag],
            'clusterings': clusterings,
            'clusterings_large': clusterings_large,
            'clusterings_merged': clusterings_merged,

        }

    ###################################################################
    # # Decide which model to use...
    ###################################################################
    selected_model_number = select_model(model_results)
    selected_model = models[selected_model_number]
    selected_model_clusterings = model_results[selected_model_number]['clusterings_merged']
    if debug:
        print(f'\tSelected model is: {selected_model_number}')

    ###################################################################
    # # Find the events
    ###################################################################
    time_event_start = time.time()
    # Calculate the shell events
    events: Events = Events.from_clusters(
        selected_model_clusterings,
        selected_model,
        dataset_xmaps,
        grid,
        alignments[test_dtag],
        max_site_distance_cutoff,
        min_bdc, max_bdc,
        None,
    )

    time_event_finish = time.time()
    dataset_log[constants.LOG_DATASET_EVENT_TIME] = time_event_finish - time_event_start
    update_log(dataset_log, dataset_log_path)

    ###################################################################
    # # Generate event maps
    ###################################################################
    time_event_map_start = time.time()

    # Save the event maps!
    # printer.pprint(events)
    events.save_event_maps(
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
        mapper=process_local_serial,
    )

    time_event_map_finish = time.time()
    dataset_log[constants.LOG_DATASET_EVENT_MAP_TIME] = time_event_map_finish - time_event_map_start
    update_log(dataset_log, dataset_log_path)

    time_dataset_finish = time.time()
    dataset_log[constants.LOG_DATASET_TIME] = time_dataset_finish - time_dataset_start
    update_log(dataset_log, dataset_log_path)

    return DatasetResult(
        dtag=test_dtag.dtag,
        events=events,
        log=dataset_log,
    )


def process_shell_multiple_models(
        shell: ShellMultipleModels,
        datasets: Dict[Dtag, Dataset],
        alignments,
        grid,
        pandda_fs_model: PanDDAFSModel,
        reference,
        process_local,
        structure_factors: StructureFactors,
        sample_rate: float,
        contour_level,
        cluster_cutoff_distance_multiplier,
        min_blob_volume,
        min_blob_z_peak,
        outer_mask,
        inner_mask_symmetry,
        max_site_distance_cutoff,
        min_bdc,
        max_bdc,
        memory_availability,
        statmaps,
        debug=False,
):
    time_shell_start = time.time()
    shell_log_path = pandda_fs_model.shell_dirs.shell_dirs[shell.res].log_path
    shell_log = {}

    # Seperate out test and train datasets
    shell_datasets: Dict[Dtag, Dataset] = {
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
    shell_working_resolution = Resolution(
        min([datasets[dtag].reflections.resolution().resolution for dtag in shell.all_dtags]))
    shell_truncated_datasets: Datasets = truncate(
        shell_datasets,
        resolution=shell_working_resolution,
        structure_factors=structure_factors,
    )
    shell_log["Shell Working Resolution"] = shell_working_resolution.resolution

    ###################################################################
    # # Generate aligned Xmaps
    ###################################################################
    time_xmaps_start = time.time()

    load_xmap_paramaterised = partial(
        Xmap.from_unaligned_dataset_c,
        grid=grid,
        structure_factors=structure_factors,
        sample_rate=sample_rate,
    )

    results = process_local(
        partial(
            load_xmap_paramaterised,
            shell_truncated_datasets[key],
            alignments[key],
        )
        for key
        in shell_truncated_datasets
    )

    xmaps = {
        dtag: xmap
        for dtag, xmap
        in zip(shell_truncated_datasets, results)
    }

    time_xmaps_finish = time.time()
    shell_log[constants.LOG_SHELL_XMAP_TIME] = time_xmaps_finish - time_xmaps_start
    update_log(shell_log, shell_log_path)

    ###################################################################
    # # Get the models to test
    ###################################################################
    models = get_models(
        shell.train_dtags,
        xmaps,
        grid,
        process_local,
    )

    ###################################################################
    # # Process each test dataset
    ###################################################################
    # Now that all the data is loaded, get the comparison set and process each test dtag
    if memory_availability == "very_low":
        process_local_in_dataset = process_local_serial
        process_local_over_datasets = process_local_serial
    elif memory_availability == "low":
        process_local_in_dataset = process_local
        process_local_over_datasets = process_local_serial
    elif memory_availability == "high":
        process_local_in_dataset = process_local_serial
        process_local_over_datasets = process_local

    process_dataset_paramaterized = partial(
        process_dataset_multiple_models,
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
        sample_rate=sample_rate,
        statmaps=statmaps,
        process_local=process_local_in_dataset,
        debug=debug,
    )

    # Process each dataset in the shell
    all_train_dtags = [_dtag for l in shell.train_dtags.values() for _dtag in l]
    # dataset_dtags = {_dtag:  for _dtag in shell.test_dtags for n in shell.train_dtags}
    dataset_dtags = {_dtag: [_dtag] + all_train_dtags for _dtag in shell.test_dtags}
    results = process_local_over_datasets(
        [
            partial(
                process_dataset_paramaterized,
                test_dtag,
                dataset_truncated_datasets={_dtag: shell_truncated_datasets[_dtag] for _dtag in
                                            dataset_dtags[test_dtag]},
                dataset_xmaps={_dtag: xmaps[_dtag] for _dtag in dataset_dtags[test_dtag]},
            )
            for test_dtag
            in shell.test_dtags
        ],
    )

    # Update shell log with dataset results
    shell_log[constants.LOG_SHELL_DATASET_LOGS] = {}
    for result in results:
        if result:
            shell_log[constants.LOG_SHELL_DATASET_LOGS][result.dtag] = result.log

    time_shell_finish = time.time()
    shell_log[constants.LOG_SHELL_TIME] = time_shell_finish - time_shell_start
    update_log(shell_log, shell_log_path)

    return ShellResult(
        shell=shell,
        dataset_results={dtag: result for dtag, result in zip(shell.train_dtags, results) if result},
        log=shell_log,
    )
