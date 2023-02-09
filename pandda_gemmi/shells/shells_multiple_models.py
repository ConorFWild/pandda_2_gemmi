from __future__ import annotations

import typing
import dataclasses

from joblib.externals.loky import set_loky_pickler

set_loky_pickler('pickle')

from typing import *

import numpy as np

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.common import Dtag
from pandda_gemmi.dataset import Dataset, Datasets, Resolution
from pandda_gemmi.comparators import ComparatorCluster


@dataclasses.dataclass()
class ShellMultipleModels:
    # number: int
    res: float
    test_dtags: typing.List[Dtag]
    train_dtags: typing.Dict[int, Set[Dtag]]
    all_dtags: typing.Set[Dtag]
    # datasets: Datasets
    # res_max: Resolution
    # res_min: Resolution


def get_shells_multiple_models_dep(
        datasets: DatasetsInterface,
        comparators: ComparatorsInterface,
        min_characterisation_datasets,
        max_shell_datasets,
        high_res_increment,
        only_datasets: Optional[List[str]],
        debug: Debug = Debug.DEFAULT
):
    # For each dataset + set of comparators, include all of these to be loaded in the set of the shell of their highest
    # Common reoslution

    # Get the dictionary of resolutions for convenience
    resolutions = {dtag: datasets[dtag].reflections.resolution().resolution for dtag in datasets}

    # Find the minimum resolutioin with enough training data
    dtags_by_resolution = [x for x in sorted(resolutions,
                                             key=lambda _dtag: resolutions[_dtag])]
    lowest_valid_res = datasets[dtags_by_resolution[min_characterisation_datasets + 1]].reflections.resolution(

    ).resolution
    if debug >= Debug.PRINT_SUMMARIES:
        print(f'\tLowest valid resolution is: {lowest_valid_res}')

    # Get the shells: start with the highest res dataset and count up in increments of high_res_increment to the
    # Lowest res dataset
    # reses = np.arange(min(resolutions.values()), max(resolutions.values()), high_res_increment)
    # reses = np.arange(min(resolutions.values()), max(resolutions.values()), high_res_increment)
    reses = np.arange(lowest_valid_res, max(resolutions.values()), high_res_increment)

    shells_test = {res: set() for res in reses}
    shells_train = {res: {} for res in reses}

    # Iterate over comparators, getting the resolution range, the lowest res in it, and then including all
    # in the set of the first shell of sufficiently low res
    # for res in reses:
    #     for cluster_num, comparator_cluster in comparators.items():
    #         shells_train[res][cluster_num] = []
    #
    #         # Sort dtags by distance to cluster
    #         sorted_distance_to_cluster = sorted(
    #             comparator_cluster.dtag_distance_to_cluster,
    #             key=lambda _dtag: comparator_cluster.dtag_distance_to_cluster[_dtag]
    #                                             )
    #
    #         # Iterate over dtags, from closest to cluster to furthest, adding those of the right resolution until
    #         # comparison set is full
    #         for dtag in sorted_distance_to_cluster:
    #             if datasets[dtag].reflections.resolution().resolution < res:
    #                 shells_train[res][cluster_num].append(dtag)
    #
    #                 # If enough datasets for training, exit loop and move onto next cluster
    #                 if len(shells_train[res][cluster_num]) >= min_characterisation_datasets:
    #                     break

    for res in reses:
        resolution_shell_dtags = {_dtag: _resolution for _dtag, _resolution in resolutions.items() if _resolution > res}
        shell_high_res_dtag = min(
            resolution_shell_dtags,
            key=lambda _dtag: resolution_shell_dtags[_dtag]
        )
        high_res_dtag_comparators = comparators[shell_high_res_dtag]
        for comparator_num, comparator_dtags in high_res_dtag_comparators.items():
            shells_train[res][comparator_num] = comparator_dtags[:min_characterisation_datasets]

    # Add each testing dtag to the appropriate shell
    for dtag in datasets:
        # Check if only_datasets is set, and if so skip any dataset not in it
        if only_datasets:
            if dtag.dtag not in only_datasets:
                continue

        # Find the first shell whose res is higher
        dtag_res: float = datasets[dtag].reflections.get_resolution()
        for res in reses:
            if res > dtag_res:
                shells_test[res] = shells_test[res].union({dtag, })

                # Make sure they only appear in one shell
                break

    # Create shells
    shells = {}
    for j, res in enumerate(reses):

        # Collect a set of all dtags
        all_dtags = set()

        # Add all the test dtags
        for dtag in shells_test[res]:
            all_dtags = all_dtags.union({dtag, })

        # Add all the train dtags
        for cluster_num, cluster_dtags in shells_train[res].items():
            all_dtags = all_dtags.union(cluster_dtags)

        # Create the shell
        shell = ShellMultipleModels(
            res,
            [x for x in shells_test[res]],
            shells_train[res],
            all_dtags,
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


def get_shells_multiple_models(
        datasets: DatasetsInterface,
        comparators: ComparatorsInterface,
        min_characterisation_datasets,
        max_shell_datasets,
        high_res_increment,
        only_datasets: Optional[List[str]],
        debug: Debug = Debug.DEFAULT
):
    # For each dataset + set of comparators, include all of these to be loaded in the set of the shell of their highest
    # Common reoslution

    # Get the dictionary of resolutions for convenience
    resolutions = {dtag: datasets[dtag].reflections.resolution().resolution for dtag in datasets}

    # Find the minimum resolutioin with enough training data
    dtags_by_resolution = [x for x in sorted(resolutions,
                                             key=lambda _dtag: resolutions[_dtag])]
    lowest_valid_res = datasets[dtags_by_resolution[min_characterisation_datasets + 1]].reflections.resolution(

    ).resolution
    if debug >= Debug.PRINT_SUMMARIES:
        print(f'\tLowest valid resolution is: {lowest_valid_res}')

    # Get the shells: start with the highest res dataset and count up in increments of high_res_increment to the
    # Lowest res dataset
    # reses = np.arange(min(resolutions.values()), max(resolutions.values()), high_res_increment)
    # reses = np.arange(min(resolutions.values()), max(resolutions.values()), high_res_increment)
    reses = np.arange(lowest_valid_res, max(resolutions.values()), high_res_increment)

    shells_test = {}
    shells_train = {}

    # Iterate over comparators, getting the resolution range, the lowest res in it, and then including all
    # in the set of the first shell of sufficiently low res
    # for res in reses:
    #     for cluster_num, comparator_cluster in comparators.items():
    #         shells_train[res][cluster_num] = []
    #
    #         # Sort dtags by distance to cluster
    #         sorted_distance_to_cluster = sorted(
    #             comparator_cluster.dtag_distance_to_cluster,
    #             key=lambda _dtag: comparator_cluster.dtag_distance_to_cluster[_dtag]
    #                                             )
    #
    #         # Iterate over dtags, from closest to cluster to furthest, adding those of the right resolution until
    #         # comparison set is full
    #         for dtag in sorted_distance_to_cluster:
    #             if datasets[dtag].reflections.resolution().resolution < res:
    #                 shells_train[res][cluster_num].append(dtag)
    #
    #                 # If enough datasets for training, exit loop and move onto next cluster
    #                 if len(shells_train[res][cluster_num]) >= min_characterisation_datasets:
    #                     break

    dtag_accumulator = []
    for _current_dtag in dtags_by_resolution:

        # Check if the new dtag makes the shell too wide or large
        new_accumlator_size = len(dtag_accumulator) + 1
        current_dataset_res = datasets[_current_dtag].reflections.get_resolution()
        if len(dtag_accumulator) != 0:
            highest_shell_res = datasets[dtag_accumulator[0]].reflections.get_resolution()
        else:
            highest_shell_res = 0.0
        new_accumulator_width = current_dataset_res - highest_shell_res

        # If not, Add the dtag to the accumulator and continue
        if ((new_accumlator_size < max_shell_datasets) & (new_accumulator_width < high_res_increment)) | (
                current_dataset_res <= lowest_valid_res):
            dtag_accumulator.append(_current_dtag)

        # If so, create a shell and empty the accumulator
        else:
            # Get the current dataset resolutions
            low_res_dtag = dtag_accumulator[-1]
            low_res = resolutions[low_res_dtag]
            low_res_dtag_comparators = comparators[low_res_dtag]

            # Test dtags are accumulator set
            shells_test[low_res] = dtag_accumulator

            # Get the comparators for the lowest res dtag in accumulator (i.e. test set with res better <= train)
            shells_train[low_res] = {}
            for comparator_num, comparator_dtags in low_res_dtag_comparators.items():
                shells_train[low_res][comparator_num] = comparator_dtags[:min_characterisation_datasets]

            # Empty the accumulator
            dtag_accumulator = [_current_dtag,]

    if len(dtag_accumulator) != 0:
        low_res_dtag = dtag_accumulator[-1]
        low_res = resolutions[low_res_dtag]
        low_res_dtag_comparators = comparators[low_res_dtag]

        # Test dtags are accumulator set
        shells_test[low_res] = dtag_accumulator

        # Get the comparators for the lowest res dtag in accumulator (i.e. test set with res better <= train)
        shells_train[low_res] = {}
        for comparator_num, comparator_dtags in low_res_dtag_comparators.items():
            shells_train[low_res][comparator_num] = comparator_dtags[:min_characterisation_datasets]

    # Create shells
    shells = {}
    for res in shells_test:

        # Collect a set of all dtags
        all_dtags = set()

        # Add all the test dtags
        for dtag in shells_test[res]:
            all_dtags = all_dtags.union({dtag, })

        # Add all the train dtags
        for cluster_num, cluster_dtags in shells_train[res].items():
            all_dtags = all_dtags.union(cluster_dtags)

        # Create the shell
        shell = ShellMultipleModels(
            res,
            [x for x in shells_test[res]],
            shells_train[res],
            all_dtags,
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


@dataclasses.dataclass()
class ShellsMultipleModels:
    shells: typing.Dict[int, ShellMultipleModels]

    @staticmethod
    def from_datasets(datasets: Datasets, min_characterisation_datasets: int,
                      max_shell_datasets: int,
                      high_res_increment: float
                      ):

        sorted_dtags = list(sorted(datasets.datasets.keys(),
                                   key=lambda dtag: datasets[dtag].reflections.resolution().resolution,
                                   ))

        train_dtags = []

        shells = {}
        shell_num = 0
        shell_dtags = []
        shell_res = datasets[sorted_dtags[-1]].reflections.resolution().resolution
        for dtag in sorted_dtags:
            res = datasets[dtag].reflections.resolution().resolution

            if (len(shell_dtags) >= max_shell_datasets) or (
                    res - shell_res >= high_res_increment):
                # Get the set of all dtags in shell
                all_dtags = list(set(shell_dtags).union(set(train_dtags)))

                # Create the shell
                shell = ShellMultipleModels(shell_num,
                                            shell_dtags,
                                            train_dtags,
                                            all_dtags,
                                            Datasets({dtag: datasets[dtag] for dtag in datasets
                                                      if dtag in shell_dtags or train_dtags}),
                                            res_max=Resolution.from_float(shell_res),
                                            res_min=Resolution.from_float(res),
                                            )

                # Add shell to dict
                shells[shell_num] = shell

                # Update iteration parameters
                shell_dtags = []
                shell_res = res
                shell_num = shell_num + 1

            # Add the next shell dtag
            shell_dtags.append(dtag)

            # Check if the characterisation set is too big and pop if so
            if len(train_dtags) >= min_characterisation_datasets:
                train_dtags = train_dtags[1:]

            # Add next train dtag
            train_dtags.append(dtag)

        return ShellsMultipleModels(shells)

    def __iter__(self):
        for shell_num in self.shells:
            yield self.shells[shell_num]
