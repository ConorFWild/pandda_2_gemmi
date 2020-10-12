from __future__ import annotations

from typing import List, Dict
from dataclasses import dataclass

import numpy as np

import gemmi


@dataclass()
class SpacegroupPython:
    spacegroup: str
    
    @staticmethod
    def from_gemmi(spacegroup: gemmi.spacegroup):
        spacegroup_name = spacegroup.xhm()
        return SpacegroupPython(spacegroup_name)
    
    def to_gemmi(self):
        return gemmi.find_spacegroup_by_name(self.spacegroup)


@dataclass()
class UnitCellPython:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    
    @staticmethod
    def from_gemmi(unit_cell: gemmi.UnitCell):
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


@dataclass()
class XmapPython:
    array: np.ndarray
    spacegroup: SpacegroupPython
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
        
    def to_gemmi(self):
        grid = gemmi.FloatGrid(*self.array.shape)
        grid.spacegroup = self.spacegroup.to_gemmi()
        grid.set_unit_cell(self.unit_cell.to_gemmi())
        
        grid_array = np.array(grid, copy=False)
        grid_array[:, :, :] = self.array[:, :, :]
        
        return grid


@dataclass
class MtzDatasetPython:
    number: int
    project_name: str
    crystal_name: str
    dataset_name: str
    wavelength: float
    
    @staticmethod
    def from_gemmi(dataset: gemmi.Dataset):
        return MtzDatasetPython(dataset.number,
                                dataset.project_name,
                                dataset.crystal_name,
                                dataset.dataset_name,
                                dataset.wavelength,
                                )
    
@dataclass
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


@dataclass
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
        datasets = [MtzDatasetPython.from_gemmi(dataset) for dataset in mtz.dataset]
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
        mtz = gemmi.mtz(with_base=False)
        mtz.title = self.mtz_title
        mtz.history = self.mtz_history
        spacegroup = self.spacegroup.to_gemmi()       
        mtz.spacegroup = spacegroup 
        unit_cell = self.unit_cell.to_gemmi()
        mtz.set_cell_for_all(unit_cell)
        
        for dataset in self.datasets:
            mtz.add_dataset(dataset.dataset_name)
            ds = mtz.dataset(dataset.number)
            ds.project_name = dataset.project_name
            ds.crystal_name = dataset.crystal_name 
            ds.wavelength = dataset.wavelength
            
        for column in self.columns:
            mtz.add_column(column.label, column.column_type, dataset_id=column.dataset_id)

        mtz.set_data(self.array)
        
        return mtz
        
        



@dataclass
class StructurePython:
    json_str: str
    
    @staticmethod
    def from_gemmi(structure: gemmi.Structure):
        json_str = structure.make_mmcif_document().as_json(mmjson=True)
        return StructurePython(json_str)
    
    def to_gemmi(self):
        cif_block = gemmi.cif.read_mmjson(self.json_str)[0]
        structure = gemmi.make_structure_from_block(cif_block)
        
        return structure