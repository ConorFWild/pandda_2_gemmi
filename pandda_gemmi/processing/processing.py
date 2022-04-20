from __future__ import annotations

# Base python
import dataclasses
import time
import pprint
from functools import partial
import os
import json
from pandda_gemmi.analyse_interface import AlignmentsInterface, DatasetsInterface, GridInterface, LoadXMapInterface, \
    PanDDAFSModelInterface, ProcessorInterface, ReferenceInterface, ShellInterface, StructureFactorsInterface

printer = pprint.PrettyPrinter()

# Scientific python libraries


## Custom Imports
from pandda_gemmi.logs import (
    summarise_array,
)

from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.processing.process_local import ProcessLocalSerial

from pandda_gemmi.pandda_functions import (
    process_local_serial,
    truncate,
    save_native_frame_zmap
)
from pandda_gemmi.python_types import *
from pandda_gemmi.common import Dtag, EventID, Partial
from pandda_gemmi.fs import PanDDAFSModel, MeanMapFile, StdMapFile
from pandda_gemmi.dataset import (StructureFactors, Dataset, Datasets,
                                  Resolution, )
from pandda_gemmi.shells import Shell
from pandda_gemmi.edalignment import Partitioning, Xmap, XmapArray
from pandda_gemmi.model import Zmap, Model, Zmaps
from pandda_gemmi.event import Event, Clusterings, Clustering, Events, get_event_mask_indicies
from pandda_gemmi.density_clustering import (
    GetEDClustering, FilterEDClusteringsSize,
    FilterEDClusteringsPeak,
    MergeEDClusterings,
)


@dataclasses.dataclass()
class DatasetResult:
    dtag: Dtag
    events: Dict[EventID, Event]
    log: Dict


@dataclasses.dataclass()
class ShellResult(ShellResultInterface):
    shell: ShellInterface
    dataset_results: DatasetResultsInterface
    log: Dict


def update_log(shell_log, shell_log_path):
    if shell_log_path.exists():
        os.remove(shell_log_path)

    with open(shell_log_path, "w") as f:
        json.dump(shell_log, f)


def process_dataset(
        test_dtag,
        shell,
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
):
    time_dataset_start = time.time()

    dataset_log_path = pandda_fs_model.processed_datasets.processed_datasets[test_dtag].log_path
    dataset_log = {}
    dataset_log[constants.LOG_DATASET_TRAIN] = [_dtag.dtag for _dtag in shell.train_dtags[test_dtag]]
    update_log(dataset_log, dataset_log_path)

    masked_xmap_array = XmapArray.from_xmaps(
        dataset_xmaps,
        grid,
    )

    masked_train_xmap_array: XmapArray = masked_xmap_array.from_dtags(
        [_dtag for _dtag in shell.train_dtags[test_dtag].union({test_dtag, })])

    ###################################################################
    # # Generate the statistical model of the dataset
    ###################################################################
    time_model_start = time.time()

    # Determine the parameters of the model to find outlying electron density
    mean_array: np.ndarray = Model.mean_from_xmap_array(masked_train_xmap_array,
                                                        )  # Size of grid.partitioning.total_mask > 0
    dataset_log[constants.LOG_DATASET_MEAN] = summarise_array(mean_array)
    update_log(dataset_log, dataset_log_path)

    sigma_is: Dict[Dtag, float] = Model.sigma_is_from_xmap_array(masked_train_xmap_array,
                                                                 mean_array,
                                                                 1.5,
                                                                 )  # size of n
    dataset_log[constants.LOG_DATASET_SIGMA_I] = {_dtag.dtag: float(sigma_i) for _dtag, sigma_i in sigma_is.items()}
    update_log(dataset_log, dataset_log_path)

    sigma_s_m: np.ndarray = Model.sigma_sms_from_xmaps(masked_train_xmap_array,
                                                       mean_array,
                                                       sigma_is,
                                                       process_local,
                                                       )  # size of total_mask > 0
    dataset_log[constants.LOG_DATASET_SIGMA_S] = summarise_array(sigma_s_m)
    update_log(dataset_log, dataset_log_path)

    model: Model = Model.from_mean_is_sms(
        mean_array,
        sigma_is,
        sigma_s_m,
        grid,
    )
    time_model_finish = time.time()
    dataset_log[constants.LOG_DATASET_MODEL_TIME] = time_model_finish - time_model_start
    update_log(dataset_log, dataset_log_path)

    # Calculate z maps
    time_z_maps_start = time.time()
    zmaps: Dict[Dtag, Zmap] = Zmaps.from_xmaps(
        model=model,
        xmaps={test_dtag: dataset_xmaps[test_dtag], },
    )
    time_z_maps_finish = time.time()
    dataset_log[constants.LOG_DATASET_Z_MAPS_TIME] = time_z_maps_finish - time_z_maps_start
    update_log(dataset_log, dataset_log_path)

    zmap = zmaps[test_dtag]

    native_grid = dataset_truncated_datasets[test_dtag].reflections.reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        # sample_rate=sample_rate,  # TODO: make this d_min/0.5?
        sample_rate=dataset_truncated_datasets[test_dtag].reflections.resolution().resolution / 0.5
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

    # for dtag in zmaps:
    #     zmap = zmaps[dtag]
    #
    #     native_grid = dataset_truncated_datasets[test_dtag].reflections.reflections.transform_f_phi_to_map(
    #         structure_factors.f,
    #         structure_factors.phi,
    #         # sample_rate=sample_rate,  # TODO: make this d_min/0.5?
    #         sample_rate=dataset_truncated_datasets[test_dtag].reflections.resolution().resolution / 0.5
    #     )
    #
    #     partitioning = Partitioning.from_structure_multiprocess(
    #         dataset_truncated_datasets[test_dtag].structure,
    #         native_grid,
    #         outer_mask,
    #         inner_mask_symmetry,
    #     )
    #     # pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save_reference_frame_zmap(zmap)
    #
    #     save_native_frame_zmap(
    #         pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.path,
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
    #
    #     if statmaps:
    #         mean_map_file = MeanMapFile.from_zmap_file(
    #             pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file)
    #         mean_map_file.save_native_frame_mean_map(
    #             model,
    #             zmap,
    #             dataset_truncated_datasets[test_dtag],
    #             alignments[test_dtag],
    #             grid,
    #             structure_factors,
    #             outer_mask,
    #             inner_mask_symmetry,
    #             partitioning,
    #             sample_rate,
    #         )
    #
    #         std_map_file = StdMapFile.from_zmap_file(pandda_fs_model.processed_datasets.processed_datasets[
    #                                                      dtag].z_map_file)
    #         std_map_file.save_native_frame_std_map(
    #             dtag,
    #             model,
    #             zmap,
    #             dataset_truncated_datasets[test_dtag],
    #             alignments[test_dtag],
    #             grid,
    #             structure_factors,
    #             outer_mask,
    #             inner_mask_symmetry,
    #             partitioning,
    #             sample_rate,
    #         )

    ###################################################################
    # # Cluster the outlying density
    ###################################################################
    time_cluster_start = time.time()

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

    dataset_log[constants.LOG_DATASET_INITIAL_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings.values()])
    # update_log(dataset_log, dataset_log_path)
    cluster_sizes = {}
    for dtag, clustering in clusterings.items():
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
    # update_log(dataset_log, dataset_log_path)

    # Filter out small clusters
    clusterings_large: EDClusteringsInterface = FilterEDClusteringsSize()(clusterings,
                                                                          grid,
                                                                          min_blob_volume,
                                                                          )
    # if debug:
    #     print("\t\tAfter filtering: large: {}".format(
    #         {dtag: len(cluster) for dtag, cluster in
    #          zip(clusterings_large, clusterings_large.values())}))
    dataset_log[constants.LOG_DATASET_LARGE_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings_large.values()])
    # update_log(dataset_log, dataset_log_path)

    # Filter out weak clusters (low peak z score)
    clusterings_peaked: EDClusteringsInterface = FilterEDClusteringsPeak()(clusterings_large,
                                                                           grid,
                                                                           min_blob_z_peak)
    # if debug:
    #     print("\t\tAfter filtering: peak: {}".format(
    #         {dtag: len(cluster) for dtag, cluster in
    #          zip(clusterings_peaked, clusterings_peaked.values())}))
    dataset_log[constants.LOG_DATASET_PEAKED_CLUSTERS_NUM] = sum(
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
    # if debug:
    #     print("\t\tAfter filtering: merged: {}".format(
    #         {dtag: len(_cluster) for dtag, _cluster in
    #          zip(clusterings_merged, clusterings_merged.values())}))
    dataset_log[constants.LOG_DATASET_MERGED_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings_merged.values()])

    # # Get the clustered electron desnity outliers
    # cluster_paramaterised = partial(
    #     Clustering.from_zmap,
    #     reference=reference,
    #     grid=grid,
    #     contour_level=contour_level,
    #     cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
    # )
    #
    # # clusterings = process_local(
    # #     [
    # #         partial(cluster_paramaterised, zmaps[dtag], )
    # #         for dtag
    # #         in zmaps
    # #     ]
    # # )
    # clusterings = [
    #         partial(cluster_paramaterised, zmaps[dtag], )()
    #         for dtag
    #         in zmaps
    #     ]
    #
    # clusterings = Clusterings({dtag: clustering for dtag, clustering in zip(zmaps, clusterings)})
    # # print("\t\tIntially found clusters: {}".format(
    # #     {
    # #         dtag: (
    # #             len(clustering),
    # #             max([len(cluster.indexes[0]) for cluster in clustering.clustering.values()] + [0, ]),
    # #             max([cluster.size(grid) for cluster in clustering.clustering.values()] + [0, ]),
    # #         )
    # #         for dtag, clustering in zip(clusterings.clusterings, clusterings.clusterings.values())
    # #     }
    # # )
    # # )
    # dataset_log[constants.LOG_DATASET_INITIAL_CLUSTERS_NUM] = sum(
    #     [len(clustering) for clustering in clusterings.clusterings.values()])
    # update_log(dataset_log, dataset_log_path)
    # cluster_sizes = {}
    # for dtag, clustering in clusterings.clusterings.items():
    #     for cluster_num, cluster in clustering.clustering.items():
    #         cluster_sizes[int(cluster_num)] = {
    #             "size": float(cluster.size(grid)),
    #             "centroid": (float(cluster.centroid[0]), float(cluster.centroid[1]), float(cluster.centroid[2])),
    #         }
    # dataset_log[constants.LOG_DATASET_CLUSTER_SIZES] = {
    #     cluster_num: cluster_sizes[cluster_num]
    #     for j, cluster_num
    #     in enumerate(sorted(
    #         cluster_sizes, key=lambda _cluster_num: cluster_sizes[_cluster_num]["size"],
    #         reverse=True,
    #     ))
    #     if j < 10
    # }
    # update_log(dataset_log, dataset_log_path)
    #
    # # Filter out small clusters
    # clusterings_large: Clusterings = clusterings.filter_size(grid,
    #                                                          min_blob_volume,
    #                                                          )
    # # print("\t\tAfter filtering: large: {}".format(
    # #     {dtag: len(cluster) for dtag, cluster in
    # #      zip(clusterings_large.clusterings, clusterings_large.clusterings.values())}))
    # dataset_log[constants.LOG_DATASET_LARGE_CLUSTERS_NUM] = sum(
    #     [len(clustering) for clustering in clusterings_large.clusterings.values()])
    # update_log(dataset_log, dataset_log_path)
    #
    # # Filter out weak clusters (low peak z score)
    # clusterings_peaked: Clusterings = clusterings_large.filter_peak(grid,
    #                                                                 min_blob_z_peak)
    # # print("\t\tAfter filtering: peak: {}".format(
    # #     {dtag: len(cluster) for dtag, cluster in
    # #      zip(clusterings_peaked.clusterings, clusterings_peaked.clusterings.values())}))
    # dataset_log[constants.LOG_DATASET_PEAKED_CLUSTERS_NUM] = sum(
    #     [len(clustering) for clustering in clusterings_peaked.clusterings.values()])
    # update_log(dataset_log, dataset_log_path)
    #
    # # Add the event masks
    # for clustering_id, clustering in clusterings_peaked.clusterings.items():
    #     for cluster_id, cluster in clustering.clustering.items():
    #         cluster.event_mask_indicies = get_event_mask_indicies(zmaps[test_dtag], cluster.cluster_positions_array)
    #
    #
    # # Merge the clusters
    # clusterings_merged = clusterings_peaked.merge_clusters()
    # # print("\t\tAfter filtering: merged: {}".format(
    # #     {dtag: len(_cluster) for dtag, _cluster in
    # #      zip(clusterings_merged.clusterings, clusterings_merged.clusterings.values())}))
    # dataset_log[constants.LOG_DATASET_MERGED_CLUSTERS_NUM] = sum(
    #     [len(clustering) for clustering in clusterings_merged.clusterings.values()])
    # update_log(dataset_log, dataset_log_path)
    #
    # # Add the event mask
    # # for clustering_id, clustering in clusterings_merged.clusterings.items():
    # #     for cluster_id, cluster in clustering.clustering.items():
    # #         cluster.event_mask_indicies = get_event_mask_indicies(zmaps[test_dtag], cluster.cluster_positions_array)

    time_cluster_finish = time.time()
    dataset_log[constants.LOG_DATASET_CLUSTER_TIME] = time_cluster_finish - time_cluster_start
    update_log(dataset_log, dataset_log_path)

    ###################################################################
    # # Find the events
    ###################################################################
    time_event_start = time.time()
    # Calculate the shell events
    events: Events = Events.from_clusters(
        clusterings_merged,
        model,
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
        model,
        pandda_fs_model,
        grid,
        structure_factors,
        outer_mask,
        inner_mask_symmetry,
        sample_rate,
        native_grid,
        mapper=ProcessLocalSerial(),
    )

    time_event_map_finish = time.time()
    dataset_log[constants.LOG_DATASET_EVENT_MAP_TIME] = time_event_map_finish - time_event_map_start
    update_log(dataset_log, dataset_log_path)

    time_dataset_finish = time.time()
    dataset_log[constants.LOG_DATASET_TIME] = time_dataset_finish - time_dataset_start
    update_log(dataset_log, dataset_log_path)

    return DatasetResult(
        dtag=test_dtag.dtag,
        events={event_id: event for event_id, event in events.events.items()},
        log=dataset_log,
    )


# Define how to process a shell
def process_shell(
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
) -> ShellResultInterface:
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
        min([datasets[dtag].reflections.get_resolution() for dtag in shell.all_dtags]))
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

    # load_xmap_paramaterised = partial(
    #     Xmap.from_unaligned_dataset_c,
    #     grid=grid,
    #     structure_factors=structure_factors,
    #     sample_rate=sample_rate,
    # )

    results = process_local(
        Partial(load_xmap_func).paramaterise(
            shell_truncated_datasets[key],
            alignments[key],
            grid=grid,
            structure_factors=structure_factors,
            sample_rate=sample_rate,
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
        process_dataset,
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
    )

    # Process each dataset in the shell
    all_train_dtags_unmerged = [_dtag for l in shell.train_dtags.values() for _dtag in l]
    all_train_dtags = []
    for _dtag in all_train_dtags_unmerged:
        if _dtag not in all_train_dtags:
            all_train_dtags.append(_dtag)

    dataset_dtags = {_dtag: [_dtag] + all_train_dtags for _dtag in shell.test_dtags}

    results = process_local_over_datasets(
        [
            Partial(
                process_dataset).paramaterise(
                test_dtag,
                dataset_truncated_datasets={_dtag: shell_truncated_datasets[_dtag] for _dtag in
                                            dataset_dtags[test_dtag]},
                dataset_xmaps={_dtag: xmaps[_dtag] for _dtag in dataset_dtags[test_dtag]},
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
            )
            for test_dtag
            in shell.train_dtags
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

    shell_result: ShellResultInterface = ShellResult(
        shell=shell,
        dataset_results={dtag: result for dtag, result in zip(shell.train_dtags, results) if result},
        log=shell_log,
    )

    return shell_result

###################################################################
# # LOW MEMORY VERSIONS
###################################################################
# def process_dataset_low_mem(
#         test_dtag,
#         shell,
#         dataset_truncated_datasets,
#         alignments,
#         pandda_fs_model,
#         reference,
#         grid,
#         contour_level,
#         cluster_cutoff_distance_multiplier,
#         min_blob_volume,
#         min_blob_z_peak,
#         structure_factors,
#         outer_mask,
#         inner_mask_symmetry,
#         max_site_distance_cutoff,
#         min_bdc,
#         max_bdc,
#         sample_rate,
#         process_local=process_local_serial,
#
# ):
#     time_dataset_start = time.time()
#
#     dataset_log = {}
#     dataset_log[constants.LOG_DATASET_TRAIN] = [_dtag.dtag for _dtag in shell.train_dtags[test_dtag]]
#
#     ###################################################################
#     # # Generate aligned Xmaps
#     ###################################################################
#     print("Loading xmaps")
#     time_xmaps_start = time.time()
#
#     load_xmap_paramaterised = partial(
#         Xmap.from_unaligned_dataset_c,
#         grid=grid,
#         structure_factors=structure_factors,
#         sample_rate=sample_rate,
#     )
#
#     results = process_local(
#         partial(
#             load_xmap_paramaterised,
#             dataset_truncated_datasets[key],
#             alignments[key],
#         )
#         for key
#         in dataset_truncated_datasets
#     )
#
#     dataset_xmaps = {
#         dtag: xmap
#         for dtag, xmap
#         in zip(dataset_truncated_datasets, results)
#     }
#
#     time_xmaps_finish = time.time()
#     print(f"Mapped {len(dataset_xmaps)} xmaps in {time_xmaps_finish - time_xmaps_start}")
#     dataset_log[constants.LOG_DATASET_XMAP_TIME] = time_xmaps_finish - time_xmaps_start
#
#     masked_xmap_array = XmapArray.from_xmaps(
#         dataset_xmaps,
#         grid,
#     )
#
#     print(f"\t\tProcessing dtag: {test_dtag}: {shell.train_dtags[test_dtag]}")
#     masked_train_xmap_array: XmapArray = masked_xmap_array.from_dtags(
#         [_dtag for _dtag in shell.train_dtags[test_dtag].union({test_dtag, })])
#
#     ###################################################################
#     # # Generate the statistical model of the dataset
#     ###################################################################
#     time_model_start = time.time()
#
#     # Determine the parameters of the model to find outlying electron density
#     print("Fitting model")
#     mean_array: np.ndarray = Model.mean_from_xmap_array(masked_train_xmap_array,
#                                                         )  # Size of grid.partitioning.total_mask > 0
#     dataset_log[constants.LOG_DATASET_MEAN] = summarise_array(mean_array)
#
#     print("fitting sigma i")
#     sigma_is: Dict[Dtag, float] = Model.sigma_is_from_xmap_array(masked_train_xmap_array,
#                                                                  mean_array,
#                                                                  1.5,
#                                                                  )  # size of n
#     dataset_log[constants.LOG_DATASET_SIGMA_I] = {_dtag.dtag: float(sigma_i) for _dtag, sigma_i in sigma_is.items()}
#
#     print("fitting sigma s m")
#     sigma_s_m: np.ndarray = Model.sigma_sms_from_xmaps(masked_train_xmap_array,
#                                                        mean_array,
#                                                        sigma_is,
#                                                        process_local,
#                                                        )  # size of total_mask > 0
#     dataset_log[constants.LOG_DATASET_SIGMA_S] = summarise_array(sigma_s_m)
#
#     model: Model = Model.from_mean_is_sms(
#         mean_array,
#         sigma_is,
#         sigma_s_m,
#         grid,
#     )
#     time_model_finish = time.time()
#     dataset_log[constants.LOG_DATASET_MODEL_TIME] = time_model_finish - time_model_start
#     print(f"Calculated model s in: {dataset_log[constants.LOG_DATASET_MODEL_TIME]}")
#
#     # Calculate z maps
#     print("Getting zmaps")
#     time_z_maps_start = time.time()
#     zmaps: Dict[Dtag, Zmap] = Zmaps.from_xmaps(
#         model=model,
#         xmaps={test_dtag: dataset_xmaps[test_dtag], },
#     )
#     time_z_maps_finish = time.time()
#     dataset_log[constants.LOG_DATASET_Z_MAPS_TIME] = time_z_maps_finish - time_z_maps_start
#
#     for dtag in zmaps:
#         zmap = zmaps[dtag]
#         pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save(zmap)
#
#     ###################################################################
#     # # Cluster the outlying density
#     ###################################################################
#     time_cluster_start = time.time()
#
#     # Get the clustered electron desnity outliers
#     print("clusting")
#     cluster_paramaterised = partial(
#         Clustering.from_zmap,
#         reference=reference,
#         grid=grid,
#         contour_level=contour_level,
#         cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
#     )
#
#     clusterings = process_local(
#         [
#             partial(cluster_paramaterised, zmaps[dtag], )
#             for dtag
#             in zmaps
#         ]
#     )
#     clusterings = Clusterings({dtag: clustering for dtag, clustering in zip(zmaps, clusterings)})
#     print("\t\tIntially found clusters: {}".format(
#         {
#             dtag: (
#                 len(clustering),
#                 max([len(cluster.indexes[0]) for cluster in clustering.clustering.values()] + [0, ]),
#                 max([cluster.size(grid) for cluster in clustering.clustering.values()] + [0, ]),
#             )
#             for dtag, clustering in zip(clusterings.clusterings, clusterings.clusterings.values())
#         }
#     )
#     )
#     dataset_log[constants.LOG_DATASET_INITIAL_CLUSTERS_NUM] = sum(
#         [len(clustering) for clustering in clusterings.clusterings.values()])
#
#     # Filter out small clusters
#     clusterings_large: Clusterings = clusterings.filter_size(grid,
#                                                              min_blob_volume,
#                                                              )
#     print("\t\tAfter filtering: large: {}".format(
#         {dtag: len(cluster) for dtag, cluster in
#          zip(clusterings_large.clusterings, clusterings_large.clusterings.values())}))
#     dataset_log[constants.LOG_DATASET_LARGE_CLUSTERS_NUM] = sum(
#         [len(clustering) for clustering in clusterings_large.clusterings.values()])
#
#     # Filter out weak clusters (low peak z score)
#     clusterings_peaked: Clusterings = clusterings_large.filter_peak(grid,
#                                                                     min_blob_z_peak)
#     print("\t\tAfter filtering: peak: {}".format(
#         {dtag: len(cluster) for dtag, cluster in
#          zip(clusterings_peaked.clusterings, clusterings_peaked.clusterings.values())}))
#     dataset_log[constants.LOG_DATASET_PEAKED_CLUSTERS_NUM] = sum(
#         [len(clustering) for clustering in clusterings_peaked.clusterings.values()])
#
#     clusterings_merged = clusterings_peaked.merge_clusters()
#     print("\t\tAfter filtering: merged: {}".format(
#         {dtag: len(_cluster) for dtag, _cluster in
#          zip(clusterings_merged.clusterings, clusterings_merged.clusterings.values())}))
#     dataset_log[constants.LOG_DATASET_MERGED_CLUSTERS_NUM] = sum(
#         [len(clustering) for clustering in clusterings_merged.clusterings.values()])
#
#     time_cluster_finish = time.time()
#     dataset_log[constants.LOG_DATASET_CLUSTER_TIME] = time_cluster_finish - time_cluster_start
#
#     ###################################################################
#     # # Find the events
#     ###################################################################
#     time_event_start = time.time()
#     # Calculate the shell events
#     print("getting events")
#     print(f"\tGot {len(clusterings_merged.clusterings)} clusters")
#     events: Events = Events.from_clusters(
#         clusterings_merged,
#         model,
#         dataset_xmaps,
#         grid,
#         alignments[test_dtag],
#         max_site_distance_cutoff,
#         min_bdc, max_bdc,
#         None,
#     )
#
#     time_event_finish = time.time()
#     dataset_log[constants.LOG_DATASET_EVENT_TIME] = time_event_finish - time_event_start
#
#     ###################################################################
#     # # Generate event maps
#     ###################################################################
#     time_event_map_start = time.time()
#
#     # Save the event maps!
#     print("print events")
#     printer.pprint(events)
#     events.save_event_maps(dataset_truncated_datasets,
#                            alignments,
#                            dataset_xmaps,
#                            model,
#                            pandda_fs_model,
#                            grid,
#                            structure_factors,
#                            outer_mask,
#                            inner_mask_symmetry,
#                            mapper=process_local,
#                            )
#
#     time_event_map_finish = time.time()
#     dataset_log[constants.LOG_DATASET_EVENT_MAP_TIME] = time_event_map_finish - time_event_map_start
#
#     time_dataset_finish = time.time()
#     dataset_log[constants.LOG_DATASET_TIME] = time_dataset_finish - time_dataset_start
#
#     return DatasetResult(
#         dtag=test_dtag.dtag,
#         events=events,
#         log=dataset_log,
#     )
#
#
# def process_shell_low_mem(
#         shell: Shell,
#         datasets: Dict[Dtag, Dataset],
#         alignments,
#         grid,
#         pandda_fs_model,
#         reference,
#         process_local,
#         structure_factors: StructureFactors,
#         sample_rate: float,
#         contour_level,
#         cluster_cutoff_distance_multiplier,
#         min_blob_volume,
#         min_blob_z_peak,
#         outer_mask,
#         inner_mask_symmetry,
#         max_site_distance_cutoff,
#         min_bdc,
#         max_bdc,
# ):
#     time_shell_start = time.time()
#     shell_log = {}
#
#     print(f"Working on shell: {shell}")
#
#     # Seperate out test and train datasets
#     shell_datasets: Dict[Dtag, Dataset] = {
#         dtag: dataset
#         for dtag, dataset
#         in datasets.items()
#         if dtag in shell.all_dtags
#     }
#
#     ###################################################################
#     # # Homogonise shell datasets by truncation of resolution
#     ###################################################################
#     print("Truncating datasets")
#     shell_truncated_datasets: Datasets = truncate(
#         shell_datasets,
#         resolution=Resolution(min([datasets[dtag].reflections.resolution().resolution for dtag in shell.all_dtags])),
#         structure_factors=structure_factors,
#     )
#
#     ###################################################################
#     # # Process each test dataset
#     ###################################################################
#     # Now that all the data is loaded, get the comparison set and process each test dtag
#     process_dataset_paramaterized = partial(
#         process_dataset_low_mem,
#         shell=shell,
#         alignments=alignments,
#         pandda_fs_model=pandda_fs_model,
#         reference=reference,
#         grid=grid,
#         contour_level=contour_level,
#         cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
#         min_blob_volume=min_blob_volume,
#         min_blob_z_peak=min_blob_z_peak,
#         structure_factors=structure_factors,
#         outer_mask=outer_mask,
#         inner_mask_symmetry=inner_mask_symmetry,
#         max_site_distance_cutoff=max_site_distance_cutoff,
#         min_bdc=min_bdc,
#         max_bdc=max_bdc,
#         sample_rate=sample_rate,
#         process_local=process_local,
#     )
#
#     # Process each dataset in the shell
#     results = process_local_serial(
#         [
#             partial(
#                 process_dataset_paramaterized,
#                 test_dtag,
#                 dataset_truncated_datasets={_dtag: shell_truncated_datasets[_dtag] for _dtag in
#                                             shell.train_dtags[test_dtag]},
#             )
#             for test_dtag
#             in shell.train_dtags
#         ],
#     )
#
#     # Update shell log with dataset results
#     shell_log[constants.LOG_SHELL_DATASET_LOGS] = {}
#     for result in results:
#         if result:
#             shell_log[constants.LOG_SHELL_DATASET_LOGS][result.dtag] = result.log
#
#     time_shell_finish = time.time()
#     shell_log[constants.LOG_SHELL_TIME] = time_shell_finish - time_shell_start
#
#     return ShellResult(
#         shell=shell,
#         dataset_results={dtag: result for dtag, result in zip(shell.train_dtags, results) if result},
#         log=shell_log,
#     )
