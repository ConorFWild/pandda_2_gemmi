from __future__ import annotations

import typing
import dataclasses

from joblib.externals.loky import set_loky_pickler

set_loky_pickler('pickle')

from typing import *

from pandda_gemmi.common import Dtag
from pandda_gemmi.dataset import Datasets, Resolution


@dataclasses.dataclass()
class Shell:
    # number: int
    res: float
    test_dtags: typing.Set[Dtag]
    train_dtags: typing.Dict[Dtag, Set[Dtag]]
    all_dtags: typing.Set[Dtag]
    # datasets: Datasets
    # res_max: Resolution
    # res_min: Resolution




@dataclasses.dataclass()
class Shells:
    shells: typing.Dict[int, Shell]

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
                shell = Shell(shell_num,
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

        return Shells(shells)

    def __iter__(self):
        for shell_num in self.shells:
            yield self.shells[shell_num]

