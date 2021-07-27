from __future__ import annotations

# Base python
import traceback
from typing import Dict, Optional, List, Tuple
import time
from pathlib import Path
import pprint
from functools import partial
import multiprocessing as mp

printer = pprint.PrettyPrinter()

# Scientific python libraries
import fire
import numpy as np
import shutil

## Custom Imports
from pandda_gemmi.logs import (
    summarise_grid, summarise_event, summarise_structure, summarise_mtz, summarise_array, save_json_log,
)
from pandda_gemmi.pandda_types import (
    PanDDAFSModel, Dataset, Datasets, Reference, Resolution,
    Grid, Alignments, Shell, Xmap, Xmaps, Zmap,
    XmapArray, Model, Dtag, Zmaps, Clustering, Clusterings,
    EventID, Event, Events, SiteTable, EventTable,
    StructureFactors, Xmap,
    DatasetResult, ShellResult,
)
from pandda_gemmi import constants
from pandda_gemmi.pandda_functions import (
    process_local_serial,
    process_local_joblib,
    process_local_multiprocessing,
    get_dask_client,
    process_global_serial,
    process_global_dask,
    get_shells,
    get_comparators_high_res_random,
    get_comparators_closest_cutoff,
    truncate,
    validate_strategy_num_datasets,
    validate,

)
from pandda_gemmi.ranking import (
    rank_events_size,
    rank_events_autobuild,
)
from pandda_gemmi.autobuild import (
    autobuild_rhofit,
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
)

def process_dataset(
        test_dtag,
        shell,
        dataset_truncated_datasets,
        alignments,
        dataset_xmaps,
        pandda_fs_model,
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
        process_local=process_local_serial,

):
    time_dataset_start = time.time()

    dataset_log = {}
    dataset_log[constants.LOG_DATASET_TRAIN] = [_dtag.dtag for _dtag in shell.train_dtags[test_dtag]]

    masked_xmap_array = XmapArray.from_xmaps(
        dataset_xmaps,
        grid,
    )

    print(f"\t\tProcessing dtag: {test_dtag}: {shell.train_dtags[test_dtag]}")
    masked_train_xmap_array: XmapArray = masked_xmap_array.from_dtags(
        [_dtag for _dtag in shell.train_dtags[test_dtag].union({test_dtag, })])

    ###################################################################
    # # Generate the statistical model of the dataset
    ###################################################################
    time_model_start = time.time()

    # Determine the parameters of the model to find outlying electron density
    print("Fitting model")
    mean_array: np.ndarray = Model.mean_from_xmap_array(masked_train_xmap_array,
                                                        )  # Size of grid.partitioning.total_mask > 0
    dataset_log[constants.LOG_DATASET_MEAN] = summarise_array(mean_array)

    print("fitting sigma i")
    sigma_is: Dict[Dtag, float] = Model.sigma_is_from_xmap_array(masked_train_xmap_array,
                                                                 mean_array,
                                                                 1.5,
                                                                 )  # size of n
    dataset_log[constants.LOG_DATASET_SIGMA_I] = {_dtag.dtag: float(sigma_i) for _dtag, sigma_i in sigma_is.items()}

    print("fitting sigma s m")
    sigma_s_m: np.ndarray = Model.sigma_sms_from_xmaps(masked_train_xmap_array,
                                                       mean_array,
                                                       sigma_is,
                                                       process_local,
                                                       )  # size of total_mask > 0
    dataset_log[constants.LOG_DATASET_SIGMA_S] = summarise_array(sigma_s_m)

    model: Model = Model.from_mean_is_sms(
        mean_array,
        sigma_is,
        sigma_s_m,
        grid,
    )
    time_model_finish = time.time()
    dataset_log[constants.LOG_DATASET_MODEL_TIME] = time_model_finish - time_model_start
    print(f"Calculated model s in: {dataset_log[constants.LOG_DATASET_MODEL_TIME]}")

    # Calculate z maps
    print("Getting zmaps")
    time_z_maps_start = time.time()
    zmaps: Dict[Dtag, Zmap] = Zmaps.from_xmaps(
        model=model,
        xmaps={test_dtag: dataset_xmaps[test_dtag], },
    )
    time_z_maps_finish = time.time()
    dataset_log[constants.LOG_DATASET_Z_MAPS_TIME] = time_z_maps_finish - time_z_maps_start

    for dtag in zmaps:
        zmap = zmaps[dtag]
        pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save(zmap)

    ###################################################################
    # # Cluster the outlying density
    ###################################################################
    time_cluster_start = time.time()

    # Get the clustered electron desnity outliers
    print("clusting")
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
    print("\t\tIntially found clusters: {}".format(
        {
            dtag: (
                len(clustering),
                max([len(cluster.indexes[0]) for cluster in clustering.clustering.values()] + [0, ]),
                max([cluster.size(grid) for cluster in clustering.clustering.values()] + [0, ]),
            )
            for dtag, clustering in zip(clusterings.clusterings, clusterings.clusterings.values())
        }
    )
    )
    dataset_log[constants.LOG_DATASET_INITIAL_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings.clusterings.values()])

    # Filter out small clusters
    clusterings_large: Clusterings = clusterings.filter_size(grid,
                                                             min_blob_volume,
                                                             )
    print("\t\tAfter filtering: large: {}".format(
        {dtag: len(cluster) for dtag, cluster in
         zip(clusterings_large.clusterings, clusterings_large.clusterings.values())}))
    dataset_log[constants.LOG_DATASET_LARGE_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings_large.clusterings.values()])

    # Filter out weak clusters (low peak z score)
    clusterings_peaked: Clusterings = clusterings_large.filter_peak(grid,
                                                                    min_blob_z_peak)
    print("\t\tAfter filtering: peak: {}".format(
        {dtag: len(cluster) for dtag, cluster in
         zip(clusterings_peaked.clusterings, clusterings_peaked.clusterings.values())}))
    dataset_log[constants.LOG_DATASET_PEAKED_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings_peaked.clusterings.values()])

    clusterings_merged = clusterings_peaked.merge_clusters()
    print("\t\tAfter filtering: merged: {}".format(
        {dtag: len(_cluster) for dtag, _cluster in
         zip(clusterings_merged.clusterings, clusterings_merged.clusterings.values())}))
    dataset_log[constants.LOG_DATASET_MERGED_CLUSTERS_NUM] = sum(
        [len(clustering) for clustering in clusterings_merged.clusterings.values()])

    time_cluster_finish = time.time()
    dataset_log[constants.LOG_DATASET_CLUSTER_TIME] = time_cluster_finish - time_cluster_start

    ###################################################################
    # # Find the events
    ###################################################################
    time_event_start = time.time()
    # Calculate the shell events
    print("getting events")
    print(f"\tGot {len(clusterings_merged.clusterings)} clusters")
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

    ###################################################################
    # # Generate event maps
    ###################################################################
    time_event_map_start = time.time()

    # Save the event maps!
    print("print events")
    printer.pprint(events)
    events.save_event_maps(dataset_truncated_datasets,
                           alignments,
                           dataset_xmaps,
                           model,
                           pandda_fs_model,
                           grid,
                           structure_factors,
                           outer_mask,
                           inner_mask_symmetry,
                           mapper=process_local_serial,
                           )

    time_event_map_finish = time.time()
    dataset_log[constants.LOG_DATASET_EVENT_MAP_TIME] = time_event_map_finish - time_event_map_start

    time_dataset_finish = time.time()
    dataset_log[constants.LOG_DATASET_TIME] = time_dataset_finish - time_dataset_start

    return DatasetResult(
        dtag=test_dtag.dtag,
        events=events,
        log=dataset_log,
    )


# Define how to process a shell
def process_shell(
        shell: Shell,
        datasets: Dict[Dtag, Dataset],
        alignments,
        grid,
        pandda_fs_model,
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
):
    time_shell_start = time.time()
    shell_log = {}

    print(f"Working on shell: {shell}")

    # Seperate out test and train datasets
    shell_datasets: Dict[Dtag, Dataset] = {
        dtag: dataset
        for dtag, dataset
        in datasets.items()
        if dtag in shell.all_dtags
    }

    ###################################################################
    # # Homogonise shell datasets by truncation of resolution
    ###################################################################
    print("Truncating datasets")
    shell_truncated_datasets: Datasets = truncate(
        shell_datasets,
        resolution=Resolution(min([datasets[dtag].reflections.resolution().resolution for dtag in shell.all_dtags])),
        structure_factors=structure_factors,
    )

    ###################################################################
    # # Generate aligned Xmaps
    ###################################################################
    print("Loading xmaps")
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
    print(f"Mapped {len(xmaps)} xmaps in {time_xmaps_finish - time_xmaps_start}")
    shell_log[constants.LOG_SHELL_XMAP_TIME] = time_xmaps_finish - time_xmaps_start

    ###################################################################
    # # Process each test dataset
    ###################################################################
    # Now that all the data is loaded, get the comparison set and process each test dtag
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
        process_local=process_local_serial,
    )

    # Process each dataset in the shell
    results = process_local(
        [
            partial(
                process_dataset_paramaterized,
                test_dtag,
                dataset_truncated_datasets={_dtag: shell_truncated_datasets[_dtag] for _dtag in
                                            shell.train_dtags[test_dtag]},
                dataset_xmaps={_dtag: xmaps[_dtag] for _dtag in shell.train_dtags[test_dtag]},
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

    return ShellResult(
        shell=shell,
        dataset_results={dtag: result for dtag, result in zip(shell.train_dtags, results) if result},
        log=shell_log,
    )
