from typing import List
import dataclasses

import numpy as np
import gemmi

from pathlib import Path

from ..interfaces import *


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
            dataset.id,
            dataset.project_name,
            dataset.crystal_name,
            dataset.dataset_name,
            dataset.wavelength,
        )


@dataclasses.dataclass()
class MtzColumnPython:
    dataset_id: int
    column_type: str
    label: str

    @staticmethod
    def from_gemmi(column: gemmi.Column):
        return MtzColumnPython(column.dataset_id,
                               column.type,
                               column.label,
                               )


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
    def from_gemmi(mtz: gemmi.Mtz):
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
    def __init__(self, path: Path, reflections):
        self.path = path
        self.reflections = reflections

    @classmethod
    def from_path(cls, path: Path):
        return cls(path, gemmi.read_mtz_file(str(path)))
