from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

import gemmi


@dataclass()
class SpacegroupPython:
    ...


@dataclass()
class UnitCellPython:
    ...


@dataclass()
class XmapPython:
    array: np.ndarray
    spacegroupo: SpacegroupPython
    unit_cell: UnitCellPython

    @staticmethod
    def from_gemmi(xmap: gemmi.FloatGrid):
        array = np.array(xmap, copy=False)
        spacegroup = SpacegroupPython.from_gemmi(xmap.spacegroup)
        unit_cell = UnitCellPython.from_gemmi(xmap.unit_cell)

        return XmapPython(array,
                          spacegroup,
                          unit_cell,
                          )


@dataclass
class MtzPython:
    array: np.ndarray
    spacegroup: SpacegroupPython
    unit_cell: UnitCellPython

    @staticmethod
    def from_gemmi(mtz: gemmi.Mtz):
        column_labels
        column_types
        spacegroup = SpacegroupPython.from_gemmi()
        unit_cell = UnitCellPython.from_gemmi(mtz.cell)



@dataclass
class StructurePython: