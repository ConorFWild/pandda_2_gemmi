from __future__ import annotations

from typing import *
import time
from functools import partial

import dataclasses

import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

from pandda_gemmi.common import Dtag
from pandda_gemmi.dataset import Dataset, Datasets, Resolution, StructureFactors
from pandda_gemmi.edalignment import Alignment, Grid, Xmap


# from pandda_gemmi.pandda_functions import truncate, from_unaligned_dataset_c_flat

def from_unaligned_dataset_c_flat(dataset: Dataset,
                                  alignment: Alignment,
                                  grid: Grid,
                                  structure_factors: StructureFactors,
                                  sample_rate: float = 3.0, ):
    xmap = Xmap.from_unaligned_dataset_c(dataset,
                                         alignment,
                                         grid,
                                         structure_factors,
                                         # sample_rate,
                                         dataset.reflections.resolution().resolution/0.5
                                         )

    xmap_array = xmap.to_array()

    masked_array = xmap_array[grid.partitioning.total_mask == 1]

    return masked_array


def truncate(datasets: Dict[Dtag, Dataset], resolution: Resolution, structure_factors: StructureFactors):
    new_datasets_resolution = {}

    # Truncate by common resolution
    for dtag in datasets:
        truncated_dataset = datasets[dtag].truncate_resolution(resolution, )

        new_datasets_resolution[dtag] = truncated_dataset

    dataset_resolution_truncated = Datasets(new_datasets_resolution)

    # Get common set of reflections
    common_reflections = dataset_resolution_truncated.common_reflections(structure_factors)

    # truncate on reflections
    new_datasets_reflections = {}
    for dtag in dataset_resolution_truncated:
        reflections = dataset_resolution_truncated[dtag].reflections.reflections
        reflections_array = np.array(reflections)

        truncated_dataset = dataset_resolution_truncated[dtag].truncate_reflections(common_reflections,
                                                                                    )
        reflections = truncated_dataset.reflections.reflections
        reflections_array = np.array(reflections)

        new_datasets_reflections[dtag] = truncated_dataset

    return new_datasets_reflections


@dataclasses.dataclass()
class ComparatorCluster:
    dtag: Dtag
    core_dtags: List[Dtag]
    core_dtags_median: np.array
    core_dtags_width: np.array
    dtag_distance_to_cluster: Dict[Dtag, float]


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
        load_xmap_paramaterised,
        debug=False
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

    # if debug:
    #     print(f'\t\tBatches are: {batches}')

    from sklearn.decomposition import PCA, IncrementalPCA
    ipca = IncrementalPCA(n_components=min(200, batch_size))

    from dask.distributed import performance_report

    for batch in batches:

        # ms = MemorySampler()
        # with performance_report(filename=f"collection_{batch}.html"):

            # if debug:
            #     print(f'\t\t\tProcessing batch: {batch}')
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
        if debug:
            # print(f'\t\t\tProcessing batch: {batch} in {finish - start}')
            print(f'\t\t\tProcessing batch in {finish - start}')


        # Get pca
        xmap_array = np.vstack([xmap for xmap in xmaps.values()])
        ipca.partial_fit(xmap_array)

        # ax = ms.plot(align=True)
        # f = plt.figure()
        # f.axes.append(ax)
        # f.savefig(f"{batch}.png")
        # plt.close('all')


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
        if debug:
            # print(f'\t\t\tProcessing batch: {batch} in {finish - start}')
            print(f'\t\t\tProcessing batch in {finish - start}')
        # Get pca
        xmap_array = np.vstack([xmap for xmap in xmaps.values()])
        transformed_arrays.append(ipca.transform(xmap_array))

    reduced_array = np.vstack(transformed_arrays)
    return reduced_array


def get_distances_between_clusters(clusters):
    num_clusters = len(clusters)
    distance_matrix = np.zeros((num_clusters, num_clusters))

    for cluster_1_num, cluster_1_median in clusters.items():
        for cluster_2_num, cluster_2_median in clusters.items():
            distance_matrix[cluster_1_num, cluster_2_num] = np.linalg.norm(cluster_2_median - cluster_1_median)

    return distance_matrix


def renumber_clusters(clusters: Dict[int, ComparatorCluster]):
    new_clusters = {}
    for j, cluster_num in enumerate(clusters):
        new_clusters[j] = clusters[cluster_num]

    return new_clusters


def refine_comparator_clusters(clusters: Dict[int, ComparatorCluster], max_comparator_sets):
    new_clusters = {}

    while len(new_clusters) > max_comparator_sets:
        new_clusters = renumber_clusters(new_clusters)

        distance_matrix = get_distances_between_clusters(new_clusters)
        closest_cluster_indexes = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        broader_cluster_index = max(
            closest_cluster_indexes,
            key=lambda index: new_clusters[index].core_dtags_width,
        )
        # del clusters[broader_cluster_index]
        new_clusters = {
            cluster_index: new_clusters[cluster_index]
            for cluster_index
            in new_clusters
            if cluster_index != broader_cluster_index
        }

    return clusters


def get_multiple_comparator_sets(
        datasets: Dict[Dtag, Dataset],
        alignments,
        grid,
        comparison_min_comparators,
        structure_factors,
        sample_rate,
        resolution_cutoff,
        process_local,
        max_comparator_sets=None,
        debug=False,
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
    if debug:
        print(f'\tFound datasets suitable for characterising clusters: {suitable_datasets}')

    dtag_list = [dtag for dtag in suitable_datasets_list]
    dtag_array = np.array(dtag_list)
    dtag_to_index = {dtag: j for j, dtag in enumerate(dtag_list)}

    # Load the xmaps
    shell_truncated_datasets: Datasets = truncate(
        suitable_datasets,
        resolution=Resolution(highest_res_datasets_max),
        structure_factors=structure_factors,
    )
    if debug:
        print('\tTruncated suitable datasets to common resolution')

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
        load_xmap_paramaterised,
        debug=debug
    )
    if debug:
        print('\tLoaded in datasets and found dimension reduced feature vectors')

    distance_matrix, clusters = get_clusters_nn(
        reduced_array,
        dtag_list,
        dtag_array,
        dtag_to_index,
    )
    if debug:
        print(f'\tFound clusters! Found {len(clusters)} clusters!')

    if max_comparator_sets:
        clusters = refine_comparator_clusters(clusters, max_comparator_sets)

    return clusters
