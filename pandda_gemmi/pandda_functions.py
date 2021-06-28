from __future__ import annotations

from typing import *

import numpy as np
import joblib

from pandda_gemmi.pandda_types import *


def process_local_joblib(n_jobs, verbosity, funcs):
    mapper = joblib.Parallel(n_jobs=n_jobs,
                             verbose=verbosity,
                             backend="loky",
                             )

    results = mapper(joblib.delayed(func)() for func in funcs)

    return results


def process_global_serial(funcs):
    results = []
    for func in funcs:
        results.append(func())

    return results


def get_comparators_high_res_random(
        datasets: Dict[Dtag, Dataset],
        comparison_min_comparators,
        comparison_max_comparators,
):
    dtag_list = [dtag for dtag in datasets]

    dtags_by_res = list(
        sorted(
            dtag_list,
            key=lambda dtag: datasets[dtag].reflections.resolution().resolution,
        )
    )

    highest_res_datasets = dtags_by_res[:comparison_min_comparators + 1]
    highest_res_datasets_max = max(
        [datasets[dtag].reflections.resolution().resolution for dtag in highest_res_datasets])

    comparators = {}
    for dtag in dtag_list:
        current_res = datasets[dtag].reflections.resolution().resolution

        truncation_res = max(current_res, highest_res_datasets_max)

        truncated_datasets = [dtag for dtag in dtag_list if
                              datasets[dtag].reflections.resolution().resolution < truncation_res]

        comparators[dtag] = list(
            np.random.choice(
                truncated_datasets,
                size=comparison_min_comparators,
                replace=False,
            )
        )

    return comparators


def get_shells(
        datasets: Dict[Dtag, Dataset],
        comparators: Dict[Dtag, List[Dtag]],
        min_characterisation_datasets,
        max_shell_datasets,
        high_res_increment,
):
    # For each dataset + set of comparators, include all of these to be loaded in the set of the shell of their highest
    # Common reoslution

    # Get the dictionary of resolutions for convenience
    resolutions = {dtag: datasets[dtag].reflections.resolution().resolution for dtag in datasets}

    # Get the shells: start with the highest res dataset and count up in increments of high_res_increment to the
    # Lowest res dataset
    reses = np.arange(min(resolutions.values()), max(resolutions.values()), high_res_increment)
    shells_test = {res: set() for res in reses}
    shells_train = {res: set() for res in reses}

    # Iterate over comparators, getting the resolution range, the lowest res in it, and then including all
    # in the set of the first shell of sufficiently low res

    for dtag, comparison_dtags in comparators.items():
        low_res = max([resolutions[comparison_dtag] for comparison_dtag in comparison_dtags])

        # Find the first shell whose res is higher
        for res in reses:
            if res > low_res:
                shells_test[res] = shells_test[res].union({dtag, })
                shells_train[res] = shells_train[res].union(set(comparison_dtags))

                # Make sure they only appear in one shell
                break

    # Create shells
    shells = {}
    for j, res in reses:
        shell = Shell(
            shells_test[res],
            shells_train[res],
            shells_test[res].union(shells_train[res]),
        )
        shells[res] = shell

    # Delete any shells that are empty
    shells_to_delete = []
    for res in reses:
        if len(shells_test[res]) == 0 or len(shells_train[res]) == 0:
            shells_to_delete.append(res)

    for res in shells_to_delete:
        del shells[res]

    return shells


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
        print(f"{dtag}")
        print(f"{reflections_array.shape}")

        truncated_dataset = dataset_resolution_truncated[dtag].truncate_reflections(common_reflections,
                                                                                    )
        reflections = truncated_dataset.reflections.reflections
        reflections_array = np.array(reflections)
        print(f"{dtag}: {reflections_array.shape}")

        new_datasets_reflections[dtag] = truncated_dataset

    return new_datasets_reflections
