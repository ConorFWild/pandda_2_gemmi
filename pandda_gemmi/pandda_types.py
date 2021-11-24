from __future__ import annotations
from types import FunctionType, MethodType

import typing
import dataclasses

import os
import time
from typing import Any, Union
import psutil
import shutil
import re
import itertools
from pathlib import Path

import numpy as np
import scipy
from scipy import spatial
from scipy import stats
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import DBSCAN

import joblib
from joblib.externals.loky import set_loky_pickler

set_loky_pickler('pickle')

from typing import *
from functools import partial

from scipy import optimize
from sklearn import neighbors

import pandas as pd
import gemmi

from pandda_gemmi.constants import *

from pandda_gemmi.python_types import *
from pandda_gemmi.common import Dtag, SiteID, EventID, EventIDX

from pandda_gemmi import pandda_exceptions

from pandda_gemmi.pandda_exceptions import *

from pandda_gemmi.dataset import StructureFactors, Structure, Reflections, Dataset, ResidueID, Datasets, \
    Resolution, Reference
from pandda_gemmi.shells import Shell
from pandda_gemmi.edalignment import Alignment, Alignments, Grid, Partitioning, Xmap, Xmaps, XmapArray
from pandda_gemmi.event import Events, Event, Clusterings, Clustering
from pandda_gemmi.sites import Sites


@dataclasses.dataclass()
class SequenceAlignment:
    _residue_id_dict: Dict[ResidueID, typing.Bool]

    def __len__(self):
        return len(self._residue_id_dict)

    def __getitem__(self, item: ResidueID):
        return self._residue_id_dict[item]

    def __iter__(self):
        for residue_id, present in self._residue_id_dict.items():
            if not present:
                continue
            else:
                yield residue_id

    def num_missing(self):
        return len([x for x in self._residue_id_dict.values() if not x])

    def num_present(self):
        return len([x for x in self._residue_id_dict.values() if x])

    def missing(self):
        return [resid for resid, present in self._residue_id_dict.items() if not present]

    def present(self):
        return [resid for resid, present in self._residue_id_dict.items() if present]

    @staticmethod
    def from_reference(reference_structure: Structure, structures: typing.Iterable[Structure]):
        residue_id_dict = {}

        for residue_id in reference_structure.protein_residue_ids():
            matched = 0
            unmatched = 0
            present = True
            # See whether other
            for dtag, structure in structures.items():
                residue_span = structure[residue_id]

                # See if residue span is empty
                if len(residue_span) == 0:
                    present = False
                    unmatched = unmatched + 1
                    print(f"Dtag: {dtag} misses residue")
                    continue

                # See if CA is present
                try:
                    ca_selection = residue_span[0]["CA"]
                except Exception as e:
                    present = False
                    unmatched = unmatched + 1
                    print(f"Dtag: {dtag} misses CA")

                    continue

                # See if can actually get CA
                try:
                    ca = ca_selection[0]
                except Exception as e:
                    present = False
                    unmatched = unmatched + 1
                    print(f"Dtag: {dtag} misses CA actually")

                    continue

                matched = matched + 1

            print((
                f"Residue: {residue_id}\n"
                f"Matched: {matched}\n"
                f"unamtched: {unmatched}"
            ))
            residue_id_dict[residue_id] = present

        return SequenceAlignment(residue_id_dict)


# @dataclasses.dataclass()
# class Shell:
#     number: int
#     test_dtags: typing.List[Dtag]
#     train_dtags: typing.List[Dtag]
#     all_dtags: typing.List[Dtag]
#     datasets: Datasets
#     res_max: Resolution
#     res_min: Resolution


@dataclasses.dataclass()
class ReferenceMap:
    dtag: Dtag
    xmap: Xmap

    @staticmethod
    def from_reference(reference: Reference, alignment: Alignment, grid: Grid, structure_factors: StructureFactors):
        xmap = Xmap.from_unaligned_dataset(reference.dataset,
                                           alignment,
                                           grid,
                                           structure_factors,
                                           )

        return ReferenceMap(reference.dtag,
                            xmap)


@dataclasses.dataclass()
class Euclidean3Coord:
    x: float
    y: float
    z: float




@dataclasses.dataclass()
class MapperJoblib:
    parallel: typing.Any

    @staticmethod
    def from_joblib(n_jobs=-1, verbose=11, max_nbytes=None, backend="multiprocessing"):
        parallel_env = joblib.Parallel(n_jobs=n_jobs, verbose=verbose,
                                       max_nbytes=max_nbytes,
                                       backend=backend,
                                       ).__enter__()

        return MapperJoblib(parallel_env)

    def map_list(self, func, *args):
        results = self.parallel(joblib.delayed(func)(*[arg[i] for arg in args])
                                for i, arg
                                in enumerate(args[0])
                                )

        return results

    def map_dict(self, func, *args):
        keys = list(args[0].keys())

        results = self.parallel(joblib.delayed(func)(*[arg[key] for arg in args])
                                for key
                                in keys
                                )

        results_dict = {keys[i]: results[i]
                        for i, key
                        in enumerate(keys)
                        }

        return results_dict


@dataclasses.dataclass()
class MapperPython:
    parallel: typing.Any

    @staticmethod
    def from_python():
        parallel_env = map
        return MapperPython(parallel_env)

    def map_list(self, func, *args):
        results = list(self.parallel(func,
                                     *args
                                     ))

        return results


@dataclasses.dataclass()
class JoblibMapper:
    mapper: Any

    @staticmethod
    def initialise():
        # mapper = joblib.Parallel(n_jobs=20, 
        #                               verbose=15,
        #                               backend="loky",
        #                                max_nbytes=None,
        #                                )
        mapper = joblib.Parallel(n_jobs=20,
                                 verbose=15,
                                 # backend="loky",
                                 backend="multiprocessing",
                                 max_nbytes=None,
                                 # prefer="threads",
                                 )
        return JoblibMapper(mapper)

    def __call__(self, iterable) -> Any:
        results = self.mapper(joblib.delayed(x)() for x in iterable)

        return results


def sample_residue(truncated_dataset: Dataset,
                   grid: Grid,
                   residue_id,
                   alignment: Alignment,
                   structure_factors: StructureFactors,
                   sample_rate: float,
                   ) -> List[float]:
    print("started")

    point_position_dict = grid.partitioning[residue_id]

    unaligned_xmap: gemmi.FloatGrid = truncated_dataset.reflections.reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )
    # Unpack the points, poitions and transforms
    point_list: List[Tuple[int, int, int]] = []
    position_list: List[Tuple[float, float, float]] = []
    transform_list: List[gemmi.transform] = []
    com_moving_list: List[np.array] = []
    com_reference_list: List[np.array] = []

    al = alignment[residue_id]
    transform = al.transform.inverse()
    com_moving = al.com_moving
    com_reference = al.com_reference

    for point, position in point_position_dict.items():
        point_list.append(point)
        position_list.append(position)
        transform_list.append(transform)
        com_moving_list.append(com_moving)
        com_reference_list.append(com_reference)

    sampled_points = gemmi.interpolate_to_list(unaligned_xmap,
                                               grid.grid,
                                               point_list,
                                               position_list,
                                               transform_list,
                                               com_moving_list,
                                               com_reference_list,
                                               )

    return np.array(sampled_points)






