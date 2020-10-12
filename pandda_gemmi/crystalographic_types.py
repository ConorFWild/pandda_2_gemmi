from __future__ import annotations
from os import stat

from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np

import gemmi

from pandda_gemmi.pandda_types import Transform, Alignment, ResidueID

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



# @dataclass
# class PositionPython:
#     ...

    
@dataclass
class PartitoningPython:
    partitioning: Dict[ResidueID, Dict[Tuple[int], Tuple[float, float, float]]]
    
    @staticmethod
    def from_gemmi(partitioning: Dict[ResidueID, Dict[Tuple[int], gemmi.Position]]):
        partitioning_dict = {}
        
        for res_id, residue_dict in partitioning.items():
            
            partitioning_dict[res_id] = {}
            
            for grid_coord, gemmi_position in residue_dict.items():
                coord_python = (gemmi_position.x,
                                gemmi_position.y,
                                gemmi_position.z,
                                )
                
                partitioning_dict[grid_coord] = coord_python
                
                
        return PartitoningPython(partitioning_dict)
                
    def to_gemmi(self):
        partitioning_dict = {}
        
        for res_id, residue_dict in self.partitioning.items():
            
            partitioning_dict[res_id] = {}
            
            for grid_coord, python_position in residue_dict.items():
                coord_python = gemmi.Position(python_position[0],
                                              python_position[1],
                                              python_position[2],
                                )
                
                partitioning_dict[grid_coord] = coord_python
                
                
        return partitioning_dict
    

@dataclass
class Int8GridPython:
    array: np.ndarray
    spacegroup: SpacegroupPython
    unit_cell: UnitCellPython

    @staticmethod
    def from_gemmi(xmap: gemmi.Int8Grid):
        array = np.array(xmap, copy=False)
        spacegroup = SpacegroupPython.from_gemmi(xmap.spacegroup)
        unit_cell = UnitCellPython.from_gemmi(xmap.unit_cell)

        return XmapPython(array,
                          spacegroup,
                          unit_cell,
                          )
        
    def to_gemmi(self):
        grid = gemmi.Int8Grid(*self.array.shape)
        grid.spacegroup = self.spacegroup.to_gemmi()
        grid.set_unit_cell(self.unit_cell.to_gemmi())
        
        grid_array = np.array(grid, copy=False)
        grid_array[:, :, :] = self.array[:, :, :]
        
        return grid

@dataclass
class FloatGridPython:
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
class TransformPython:
    transform: gemmi.Transform
    com_reference: np.array
    com_moving: np.array
    
    @staticmethod
    def from_gemmi(transform_gemmi):
        transform_python = gemmi.transform.mat.tolist()
        return TransformPython(transform_python,
                               transform_gemmi.com_reference,
                               transform_gemmi.com_moving,
                               )
        
    def to_gemmi(self):
        transform_gemmi = gemmi.Transform()
        transform_gemmi.mat.fromlist(self.transform)
        return Transform(transform_gemmi,
                         self.com_reference,
                         self.com_moving,
                         )
        
@dataclass
class AlignmentPython:
    alignment: Dict[ResidueID, TransformPython]
    
    @staticmethod
    def from_gemmi(alignment: Alignment):
        alignment_python = {}
        for res_id, transform in alignment.transforms.items():
            transform_python = TransformPython.from_gemmi(transform)
            alignment_python[res_id] = transform_python
            
        return AlignmentPython(alignment_python)
       
    def to_gemmi(self):
        alignment_gemmi = {}
        for res_id, transform in self.alignment.items():
            transform_gemmi = transform.to_gemmi()
            alignment_gemmi[res_id] = transform_gemmi
            
        return alignment_gemmi