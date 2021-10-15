from __future__ import annotations

from typing import *
from time import sleep
from functools import partial
import json
import pickle
import secrets

import numpy as np
import multiprocessing as mp
import joblib
from scipy import spatial as spsp, cluster as spc

from sklearn import decomposition, metrics
import umap
from bokeh.plotting import ColumnDataSource, figure, output_file, show, save
import hdbscan
from matplotlib import pyplot as plt

from pandda_gemmi.pandda_types import *
from pandda_gemmi import constants
from pandda_gemmi.pandda_functions import (
    truncate,
    save_plot_pca_umap_bokeh
)


def from_unaligned_dataset_c(dataset: Dataset,
                             alignment: Alignment,
                             grid: Grid,
                             structure_factors: StructureFactors,
                             sample_rate: float = 3.0, ):
    xmap = Xmap.from_unaligned_dataset_c(dataset,
                                         alignment,
                                         grid,
                                         structure_factors,
                                         sample_rate,
                                         )

    xmap_array = xmap.to_array()

    xmap_array[grid.partitioning.total_mask == 0] = 0

    return xmap_array


def get_index_dict(grid: Grid):
    partitioning: Partitioning = grid.partitioning

    index_array_dict = {}

    for resid, index_to_pos in partitioning.partitioning.items():
        index_array = np.vstack([np.array(indexes).reshape((1, 3)) for indexes in index_to_pos.keys()])

        index_tuples = (index_array[:, 0], index_array[:, 1], index_array[:, 2])

        index_array_dict[resid] = index_tuples

    return index_array_dict


def get_batches(total_sample_size, batch_size):
    tmp_batches = {}
    j = 1
    while True:
        print(f"\tJ is: {j}")
        new_batches = np.array_split(np.arange(total_sample_size), j)
        print(f"\t\tlen of new batches is {len(new_batches)}")
        tmp_batches[j] = new_batches
        j = j + 1

        if any(len(batch) < batch_size for batch in new_batches):
            batches = tmp_batches[j - 2]
            break
        else:
            print("\t\tAll batches larger than batch size, trying smaller split!")
            continue
    print(f"Batches are:")
    print(batches)

    return batches


def get_correlation(density_1, density_2):
    # demeaned_1 = density_1 - np.mean(density_1)
    # demeaned_2 = density_2 - np.mean(density_2)
    # nominator = np.sum(demeaned_1*demeaned_2)
    # denominator = np.sqrt(np.sum(np.square(density_1))*np.sum(np.square(density_2)))

    nominator = np.cov(density_1, density_2)[0, 1]
    denominator = np.sqrt(np.var(density_1) * np.var(density_2))

    return nominator / denominator


def get_comparators_local(
        reference: Reference,
        datasets: Dict[Dtag, Dataset],
        alignments,
        grid,
        comparison_min_comparators,
        comparison_max_comparators,
        structure_factors,
        sample_rate,
        resolution_cutoff,
        pandda_fs_model: PanDDAFSModel,
        process_local,
):
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
        [datasets[dtag].reflections.resolution().resolution for dtag in highest_res_datasets])

    # Load the xmaps
    print("Truncating datasets...")
    shell_truncated_datasets: Datasets = truncate(
        datasets,
        resolution=Resolution(highest_res_datasets_max),
        structure_factors=structure_factors,
    )

    # Generate aligned xmaps
    print("Loading xmaps")

    load_xmap_paramaterised = partial(
        from_unaligned_dataset_c,
        grid=grid,
        structure_factors=structure_factors,
        sample_rate=sample_rate,
    )

    # Get reduced array
    total_sample_size = len(shell_truncated_datasets)
    print(f"Total sample size = {total_sample_size}")
    batch_size = min(90, total_sample_size)
    print(f"Batch size is: {batch_size}")
    num_batches = (total_sample_size // batch_size) + 1
    print(f"Num batches is: {num_batches}")

    # Get batches
    batches = get_batches(total_sample_size, batch_size)

    # Get indexes
    partitioning_index_dict = get_index_dict(grid)

    data = {_residue_id: {} for _residue_id in partitioning_index_dict}
    print("Fitting!")
    for batch in batches:
        print(f"\tLoading dtags: {dtag_array[batch]}")
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
        print("Got xmaps!")

        # Get the maps as arrays
        print("Getting xmaps as arrays")
        xmaps = {dtag: xmap
                 for dtag, xmap
                 in zip(dtag_list, results)
                 }

        finish = time.time()
        print(f"Mapped in {finish - start}")

        for residue_id, indexes in partitioning_index_dict.items():
            data[residue_id].update({_dtag: xmap[indexes] for _dtag, xmap in zip(dtag_array[batch], xmaps.values())})

    rsccs = {}
    start = time.time()
    for residue_id, dtag_to_density_dict in data.items():
        rsccs[residue_id] = np.zeros((len(dtag_to_density_dict), len(dtag_to_density_dict)))
        for j, dtag_1 in enumerate(dtag_to_density_dict):
            density_1 = dtag_to_density_dict[dtag_1]
            for k, dtag_2 in enumerate(dtag_to_density_dict):
                density_2 = dtag_to_density_dict[dtag_2]
                rsccs[residue_id][j, k] = get_correlation(density_1, density_2)
    finish = time.time()
    print(f'Got rsccs in {finish - start}')

    for resid, rscc_matrix in rsccs.items():
        print(f"Reduced array shape: {rscc_matrix.shape}")

        # Save a bokeh plot
        labels = [dtag.dtag for dtag in dtag_list]
        known_apos = [dtag.dtag for dtag in dtag_list]

        print(f"Labels are: {labels}")
        print(f"Known apos are: {known_apos}")

        save_plot_pca_umap_bokeh(
            rscc_matrix,
            labels,
            known_apos,
            pandda_fs_model.pandda_dir / f"pca_umap_{resid.model}_{resid.chain}_{resid.insertion}.html",
        )

    exit()

    return comparators
