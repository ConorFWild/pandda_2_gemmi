from typing import List
import dataclasses

import numpy as np
import gemmi
import pandas as pd

from pathlib import Path

from ..interfaces import *
from .. import constants


@dataclasses.dataclass()
class SpacegroupPython:
    spacegroup: str

    @staticmethod
    def from_gemmi(spacegroup: SpacegroupInterface):
        spacegroup_name = spacegroup.xhm()
        return SpacegroupPython(spacegroup_name)

    def to_gemmi(self):
        return gemmi.find_spacegroup_by_name(self.spacegroup)


@dataclasses.dataclass()
class UnitCellPython:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    @staticmethod
    def from_gemmi(unit_cell: UnitCellInterface):
        return UnitCellPython(unit_cell.a,
                              unit_cell.b,
                              unit_cell.c,
                              unit_cell.alpha,
                              unit_cell.beta,
                              unit_cell.gamma,
                              )

    def to_gemmi(self):
        return gemmi.UnitCell(self.a,
                              self.b,
                              self.c,
                              self.alpha,
                              self.beta,
                              self.gamma,
                              )


@dataclasses.dataclass()
class MtzDatasetPython:
    id: int
    project_name: str
    crystal_name: str
    dataset_name: str
    wavelength: float

    @staticmethod
    def from_gemmi(mtz_dataset):
        return MtzDatasetPython(
            mtz_dataset.id,
            mtz_dataset.project_name,
            mtz_dataset.crystal_name,
            mtz_dataset.dataset_name,
            mtz_dataset.wavelength,
        )


@dataclasses.dataclass()
class MtzColumnPython:
    dataset_id: int
    column_type: str
    label: str

    @staticmethod
    def from_gemmi(column):
        return MtzColumnPython(column.dataset_id,
                               column.type,
                               column.label,
                               )


def drop_columns(reflections, f, phi):
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = reflections.spacegroup
    new_reflections.set_cell_for_all(reflections.cell)

    # Add dataset
    new_reflections.add_dataset("truncated")

    free_flag = None

    for column in reflections.columns:
        if column.label == "FREE":
            free_flag = "FREE"
            break
        if column.label == "FreeR_flag":
            free_flag = "FreeR_flag"
            break
        if column.label == "R-free-flags":
            free_flag = "R-free-flags"
            break

    if not free_flag:
        raise Exception("No RFree Flag found!")

    # Add columns
    for column in reflections.columns:
        if column.label in ["H", "K", "L", free_flag, f, phi]:
            new_reflections.add_column(column.label, column.type)

    # Get data
    data_array = np.array(reflections, copy=True)
    data = pd.DataFrame(data_array,
                        columns=reflections.column_labels(),
                        )
    data.set_index(["H", "K", "L"], inplace=True)

    # Truncate by columns
    data_indexed = data[[free_flag, f, phi]]

    # To numpy
    data_dropped_array = data_indexed.to_numpy()

    # new data
    new_data = np.hstack([data_indexed.index.to_frame().to_numpy(),
                          data_dropped_array,
                          ]
                         )

    # Update
    new_reflections.set_data(new_data)

    # Update resolution
    new_reflections.update_reso()

    return new_reflections


@dataclasses.dataclass()
class MtzPython:
    mtz_title: str
    mtz_history: str
    array: np.ndarray
    spacegroup: SpacegroupPython
    unit_cell: UnitCellPython
    datasets: List
    columns: List

    @staticmethod
    def from_gemmi(mtz):
        mtz_title = mtz.title
        mtz_history = mtz.history
        array = np.array(mtz, copy=True)
        datasets = [MtzDatasetPython.from_gemmi(dataset) for dataset in mtz.datasets]
        columns = [MtzColumnPython.from_gemmi(column) for column in mtz.columns]

        spacegroup = SpacegroupPython.from_gemmi(mtz.spacegroup)
        unit_cell = UnitCellPython.from_gemmi(mtz.cell)

        return MtzPython(mtz_title,
                         mtz_history,
                         array,
                         spacegroup,
                         unit_cell,
                         datasets,
                         columns,
                         )

    def to_gemmi(self):
        mtz = gemmi.Mtz(with_base=False)
        mtz.title = self.mtz_title
        mtz.history = self.mtz_history
        spacegroup = self.spacegroup.to_gemmi()
        mtz.spacegroup = spacegroup
        unit_cell = self.unit_cell.to_gemmi()
        mtz.set_cell_for_all(unit_cell)

        for dataset in self.datasets:
            mtz.add_dataset(dataset.dataset_name)
            ds = mtz.dataset(dataset.id)
            ds.project_name = dataset.project_name
            ds.crystal_name = dataset.crystal_name
            ds.wavelength = dataset.wavelength

        for column in self.columns:
            mtz.add_column(column.label, column.column_type, dataset_id=column.dataset_id)

        mtz.set_data(self.array)

        mtz.update_reso()

        return mtz


class Reflections(ReflectionsInterface):
    def __init__(self, path: Path, f: str, phi: str, reflections):
        self.path = path
        self.reflections = reflections
        self.f = f
        self.phi = phi

    def resolution(self):
        return self.reflections.resolution_high()

    @classmethod
    def from_path(cls, path: Path):
        reflections = gemmi.read_mtz_file(str(path))
        f, phi = cls.get_structure_factors(reflections)

        reflections = drop_columns(reflections, f, phi)
        return cls(path, f, phi, reflections)

    @classmethod
    def get_structure_factors(cls, reflections):
        column_labels = reflections.column_labels()
        for common_f_phi_label_pair in constants.COMMON_F_PHI_LABEL_PAIRS:

            f_label = common_f_phi_label_pair[0]
            phi_label = common_f_phi_label_pair[1]
            if f_label in column_labels:
                if phi_label in column_labels:
                    return f_label, phi_label

        return None, None

    def transform_f_phi_to_map(self, sample_rate: float = 3.0):
        return self.reflections.transform_f_phi_to_map(self.f, self.phi, sample_rate=sample_rate)

    def __getstate__(self):
        return (MtzPython.from_gemmi(self.reflections), self.path, self.f, self.phi)

    def __setstate__(self, data: Tuple[MtzPython, Path, str, str]):
        reflections_python = data[0]
        path = data[1]
        reflections = reflections_python.to_gemmi()
        self.reflections = reflections
        self.path = path
        self.f = data[2]
        self.phi = data[3]