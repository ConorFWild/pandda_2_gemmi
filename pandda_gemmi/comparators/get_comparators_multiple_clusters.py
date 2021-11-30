from __future__ import annotations

import os
from typing import *
import time
from time import sleep
from functools import partial
import pickle
import secrets
import dataclasses

import numpy as np
import multiprocessing as mp
import joblib
from scipy import spatial as spsp, cluster as spc
from sklearn import decomposition, metrics
import umap
from bokeh.plotting import ColumnDataSource, figure, output_file, show, save
from matplotlib import pyplot as plt
import gemmi

from pandda_gemmi import constants
from pandda_gemmi.common import Dtag
from pandda_gemmi.dataset import StructureFactors, Dataset, Datasets, Resolution
from pandda_gemmi.fs import PanDDAFSModel
from pandda_gemmi.shells import Shell
from pandda_gemmi.edalignment import Alignment, Grid, Xmap, Partitioning
from pandda_gemmi.model import Model, Zmap
from pandda_gemmi.event import Event


@dataclasses.dataclass()
class ComparatorCluster:
    dtag: Dtag
    core_dtags: List[Dtag]
    core_dtags_median: np.array
    core_dtags_width: np.array
    dtag_distance_to_cluster: Dict[Dtag,  float]


def get_clusters_nn(
        reduced_array,
        dtag_list,
        dtag_array,
        dtag_to_index,
):
    # Get distance matrix
    distance_matrix = metrics.pairwise_distances(reduced_array)

    # Get the n nearest neighbours for each point
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=30).fit(reduced_array)
    distances, indices = nbrs.kneighbors(reduced_array)

    # Get neighbourhood radii
    radii = {}
    for j, row in enumerate(distances):
        mean_distance = np.mean(row)
        radii[j] = mean_distance

    # Sort datasets by radii
    radii_sorted = {index: radii[index] for index in sorted(radii, key=lambda _index: radii[_index])}

    # Loop over datasets from narrowest to broadest, checking whether any of their neighbours have been claimed
    # If so, skip to next
    claimed = []
    cluster_num = 0
    clusters_dict: Dict[int, List[Dtag]] = {}
    cluster_leader_dict = {}
    cluster_widths = {}
    for index in radii_sorted:
        nearest_neighbour_indexes = indices[index, :]

        if not any([(j in claimed) for j in nearest_neighbour_indexes]):
            # CLaim indicies
            for j in nearest_neighbour_indexes:
                claimed.append(j)

            clusters_dict[cluster_num] = [dtag_array[j] for j in nearest_neighbour_indexes]

            cluster_leader_dict[cluster_num] = dtag_array[index]
            cluster_widths[cluster_num] = radii_sorted[index]

            cluster_num += 1

    # Get cluster medians
    cluster_medians = {}
    for cluster, cluster_dtags in clusters_dict.items():
        cluster_leader_dtag = cluster_leader_dict[cluster]
        cluster_leader_index = dtag_to_index[cluster_leader_dtag]
        cluster_medians[cluster] = reduced_array[cluster_leader_index, :].reshape((1, -1))

    # Get dtag cluster distance
    dtag_distance_to_cluster = {}
    for _dtag in dtag_list:
        dtag_index = dtag_to_index[_dtag]
        dtag_distance_to_cluster[_dtag] = {}
        dtag_coord = reduced_array[dtag_index, :].reshape((1, -1))
        for cluster, cluster_median in cluster_medians.items():
            assert cluster_median.shape[0] == dtag_coord.shape[0]
            assert cluster_median.shape[1] == dtag_coord.shape[1]
            distance = np.linalg.norm(cluster_median - dtag_coord)

            dtag_distance_to_cluster[_dtag][cluster] = distance

    # Save a bokeh plot
    labels = [dtag.dtag for dtag in dtag_list]
    # known_apos = [dtag.dtag for dtag, dataset in datasets.items() if any(dtag in x for x in cluster_cores.values())]
    known_apos = []
    for cluster_num, cluster_dtags in clusters_dict.items():
        for cluster_core_dtag in cluster_dtags:
            known_apos.append(cluster_core_dtag.dtag)

    clusters = {
        cluster_num: ComparatorCluster(
            clusters_dict[cluster_num][0],
            clusters_dict[cluster_num],
            cluster_medians[cluster_num],
            cluster_widths[cluster_num],
            {dtag: dtag_distance_to_cluster[dtag][cluster_num] for dtag in dtag_distance_to_cluster}
        )
        for cluster_num
        in clusters_dict
    }

    return distance_matrix, clusters


def get_reduced_array(
        shell_truncated_datasets,
        alignments,
        process_local,
        dtag_array,
        dtag_list,

):
    # Get reduced array
    total_sample_size = len(shell_truncated_datasets)
    batch_size = min(90, total_sample_size)
    num_batches = (total_sample_size // batch_size) + 1
    # batches = [
    #     np.arange(x*batch_size, min((x+1)*batch_size, total_sample_size))
    #     for x
    #     in range(0, num_batches)]
    tmp_batches = {}
    j = 1
    while True:
        new_batches = np.array_split(np.arange(total_sample_size), j)
        tmp_batches[j] = new_batches
        j = j + 1

        if any(len(batch) < batch_size for batch in new_batches):
            batches = tmp_batches[j - 2]
            break
        else:
            print("\t\tAll batches larger than batch size, trying smaller split!")
            continue

    from sklearn.decomposition import PCA, IncrementalPCA
    ipca = IncrementalPCA(n_components=min(200, batch_size))

    for batch in batches:
        start = time.time()
        results = process_local(
            [
                partial(
                    load_xmap_paramaterised,
                    shell_truncated_datasets[key],
                    alignments[key],
                )
                for key
                in dtag_array[batch]
            ]
        )

        # Get the maps as arrays
        xmaps = {dtag: xmap
                 for dtag, xmap
                 in zip(dtag_list, results)
                 }

        finish = time.time()

        # Get pca
        xmap_array = np.vstack([xmap for xmap in xmaps.values()])
        ipca.partial_fit(xmap_array)

    # Transform
    transformed_arrays = []
    for batch in batches:
        start = time.time()
        results = process_local(
            [
                partial(
                    load_xmap_paramaterised,
                    shell_truncated_datasets[key],
                    alignments[key],
                )
                for key
                in dtag_array[batch]
            ]
        )

        # Get the maps as arrays
        xmaps = {dtag: xmap
                 for dtag, xmap
                 in zip(dtag_list, results)
                 }

        finish = time.time()

        # Get pca
        xmap_array = np.vstack([xmap for xmap in xmaps.values()])
        transformed_arrays.append(ipca.transform(xmap_array))

    reduced_array = np.vstack(transformed_arrays)
    return reduced_array


def get_multiple_comparator_sets(
        datasets: Dict[Dtag, Dataset],
        alignments,
        grid,
        comparison_min_comparators,
        structure_factors,
        sample_rate,
        resolution_cutoff,
        process_local,
) -> Dict[int, ComparatorCluster]:
    dtag_list = [dtag for dtag in datasets]
    dtag_array = np.array(dtag_list)
    dtag_to_index = {dtag: j for j, dtag in enumerate(dtag_list)}

    dtags_by_res = list(
        sorted(
            dtag_list,
            key=lambda dtag: datasets[dtag].reflections.resolution().resolution,
        )
    )

    highest_res_datasets = dtags_by_res[:comparison_min_comparators + 1]
    highest_res_datasets_max = max(
        [
            datasets[dtag].reflections.resolution().resolution
            for dtag
            in highest_res_datasets
        ]
    )

    # Get the datasets below the upper cutoff for manifold characterisation
    suitable_datasets_list = [
        dtag for dtag in dtags_by_res if datasets[dtag].reflections.resolution().resolution < resolution_cutoff
    ]
    suitable_datasets = {dtag: dataset for dtag, dataset in datasets.items() if dtag in suitable_datasets_list}

    # Load the xmaps
    shell_truncated_datasets: Datasets = truncate(
        suitable_datasets,
        resolution=Resolution(highest_res_datasets_max),
        structure_factors=structure_factors,
    )

    # Generate aligned xmaps
    load_xmap_paramaterised = partial(
        from_unaligned_dataset_c_flat,
        grid=grid,
        structure_factors=structure_factors,
        sample_rate=sample_rate,
    )

    reduced_array = get_reduced_array(
        shell_truncated_datasets,
        alignments,
        process_local,
        dtag_array,
        dtag_list,
    )

    distance_matrix, clusters = get_clusters_nn(
        reduced_array,
        dtag_list,
        dtag_array,
        dtag_to_index,
    )

    return clusters
