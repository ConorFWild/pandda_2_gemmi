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


from sklearn import neighbors

import pandas as pd
import gemmi

from pandda_gemmi.constants import *

from pandda_gemmi.python_types import *

from pandda_gemmi.pandda_exceptions import *

@dataclasses.dataclass()
class DelayedFuncReady:
    func: MethodType
    args: Any
    
    def __call__(self) -> Any:
        return self.func(*self.args)

@dataclasses.dataclass()
class DelayedFuncWaiting:
    func: MethodType
    
    def __call__(self, *args: Any) -> Any:
        return DelayedFuncReady(self.func, args)
        
def delayed(func: MethodType):
    return DelayedFuncWaiting(func)

@dataclasses.dataclass()
class Dtag:
    dtag: str

    def __hash__(self):
        return hash(self.dtag)

    def __eq__(self, other):
        try:
            if self.dtag == other.dtag:
                return True
            else:
                return False
        except Exception as e:
            return False


@dataclasses.dataclass()
class EventIDX:
    event_idx: int

    def __hash__(self):
        return hash(self.event_idx)


@dataclasses.dataclass()
class ResidueID:
    model: str
    chain: str
    insertion: str

    @staticmethod
    def from_residue_chain(model: gemmi.Model, chain: gemmi.Chain, res: gemmi.Residue):
        return ResidueID(model.name,
                         chain.name,
                         str(res.seqid.num),
                         )
    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return ((self.model, self.chain, self.insertion) ==
                    (other.model, other.chain, other.insertion))
        return NotImplemented
    
    def __hash__(self):
        return hash((self.model, self.chain, self.insertion))
    



@dataclasses.dataclass()
class RFree:
    rfree: float

    @staticmethod
    def from_structure(structure: Structure):
        rfree = structure.structure.make_mmcif_document()[0].find_loop("_refine.ls_R_factor_R_free")[0]

        # regex = "REMARK   3   FREE R VALUE                     :  ([^\s]+))"
        # matches = re.findall(regex,
        #                      string,
        #                      )
        #
        # rfree = float(matches[0])

        return RFree(float(rfree))

    def to_float(self):
        return self.rfree


@dataclasses.dataclass()
class Structure:
    structure: gemmi.Structure
    path: typing.Union[Path, None] = None

    @staticmethod
    def from_file(file: Path) -> Structure:
        structure = gemmi.read_structure(str(file))
        structure.setup_entities()
        return Structure(structure, file)

    def rfree(self):
        return RFree.from_structure(self)

    def __getitem__(self, item: ResidueID):
        return self.structure[item.model][item.chain][item.insertion]

    # def residue_ids(self):
    #     residue_ids = []
    #     for model in self.structure:
    #         for chain in model:
    #             for residue in chain.get_polymer():
    #                 resid = ResidueID.from_residue_chain(model, chain, residue)
    #                 residue_ids.append(resid)

    #     return residue_ids

    def protein_residue_ids(self):
        for model in self.structure:
            for chain in model:
                for residue in chain.get_polymer():
                    
                    if residue.name.upper() not in RESIDUE_NAMES:
                        continue
                    
                    resid = ResidueID.from_residue_chain(model, chain, residue)                        
                    yield resid

    def protein_atoms(self):
        for model in self.structure:
            for chain in model:
                for residue in chain.get_polymer():
                    
                    if residue.name.upper() not in RESIDUE_NAMES:
                        continue
                    
                    for atom in residue:
                        yield atom

    def all_atoms(self):
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        yield atom

                        
    def align_to(self, other: Structure):
        # Warning: inplace!
        # Aligns structures usings carbon alphas and transform self into the frame of the other
        
        transform = self.get_alignment(other)
        
        # Transform positions
        for atom in self.all_atoms():
            atom.pos = transform.apply_inverse(atom.pos)
                    
        return self
    
    def get_alignment(self, other: Structure):
        # alignment returned is FROM other TO self
        
        ca_self = []
        ca_other = []
        
        # Get CAs
        for model in self.structure:
            for chain in model:
                for res_self in chain.get_polymer():
                    if res_self.name.upper() not in RESIDUE_NAMES:
                        continue
                                
                    current_res_id = ResidueID.from_residue_chain(model, chain, res_self)

                    res_other = other.structure[current_res_id][0]
                    
                    self_ca_pos = res_self["CA"][0].pos
                    other_ca_pos = res_other["CA"][0].pos
                    
                    ca_list_self = Transform.pos_to_list(self_ca_pos)
                    ca_list_other = Transform.pos_to_list(other_ca_pos)
                    
                    ca_self.append(ca_list_self)
                    ca_other.append(ca_list_other)
                    
        # Make coord matricies
        matrix_self = np.array(ca_self)
        matrix_other = np.array(ca_other)

        # Find means
        mean_self = np.mean(matrix_self, axis=0)
        mean_other = np.mean(matrix_other, axis=0)

        # demaen
        de_meaned_self = matrix_self - mean_self
        de_meaned_other = matrix_other - mean_other

        # Align
        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned_self, 
                                                                        de_meaned_other,
                                                                        )
        
        # Get transform
        vec = np.array([0.0, 0.0, 0.0])
        # Transform is from other frame to self frame
        transform = Transform.from_translation_rotation(vec,
                                                        rotation,
                                                        mean_other, 
                                                        mean_self, 
                                                        )

        return transform


    
    def get_alignment(self, other: Structure):
        # alignment returned is FROM other TO self
        
        ca_self = []
        ca_other = []
        
        # Get CAs
        for model in self.structure:
            for chain in model:
                for res_self in chain.get_polymer():      
                    if res_self.name.upper() not in RESIDUE_NAMES:
                        continue      
                    current_res_id = ResidueID.from_residue_chain(model, chain, res_self)

                    res_other = other.structure[current_res_id][0]
                    
                    self_ca_pos = res_self["CA"][0].pos
                    other_ca_pos = res_other["CA"][0].pos
                    
                    ca_list_self = Transform.pos_to_list(self_ca_pos)
                    ca_list_other = Transform.pos_to_list(other_ca_pos)
                    
                    ca_self.append(ca_list_self)
                    ca_other.append(ca_list_other)
                    
        # Make coord matricies
        matrix_self = np.array(ca_self)
        matrix_other = np.array(ca_other)

        # Find means
        mean_self = np.mean(matrix_self, axis=0)
        mean_other = np.mean(matrix_other, axis=0)

        # demaen
        de_meaned_self = matrix_self - mean_self
        de_meaned_other = matrix_other - mean_other

        # Align
        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned_self, 
                                                                        de_meaned_other,
                                                                        )
        
        # Get transform
        vec = np.array([0.0, 0.0, 0.0])
        # Transform is from other frame to self frame
        transform = Transform.from_translation_rotation(vec,
                                                        rotation,
                                                        mean_other, 
                                                        mean_self, 
                                                        )

        return transform

    def align_to(self, other: Structure):
        # Warning: inplace!
        # Aligns structures usings carbon alphas and transform self into the frame of the other
        
        transform = self.get_alignment(other)
        
        # Transform positions
        for atom in self.all_atoms():
            atom.pos = transform.apply_reference_to_moving(atom.pos)
                    
        return self                        
                        
    def __getstate__(self):
        structure_python = StructurePython.from_gemmi(self.structure)
        return (structure_python, self.path)
    
    def __setstate__(self, data: Tuple[StructurePython, Path]):
        structure_python = data[0]
        path = data[1]
        self.structure = structure_python.to_gemmi()
        self.structure.setup_entities()
        self.path = path

        


@dataclasses.dataclass()
class StructureFactors:
    f: str
    phi: str

    @staticmethod
    def from_string(string: str):
        factors = string.split(",")
        assert len(factors) == 2
        return StructureFactors(f=factors[0],
                                phi=factors[1],
                                )


@dataclasses.dataclass()
class Reflections:
    reflections: gemmi.Mtz
    path: typing.Union[Path, None] = None

    @staticmethod
    def from_file(file: Path) -> Reflections:
        reflections = gemmi.read_mtz_file(str(file))
        return Reflections(reflections, file)

    def resolution(self) -> Resolution:
        return Resolution.from_float(self.reflections.resolution_high())

    def truncate_resolution(self, resolution: Resolution) -> Reflections:
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = self.reflections.spacegroup
        new_reflections.set_cell_for_all(self.reflections.cell)

        # Add dataset
        new_reflections.add_dataset("truncated")

        # Add columns
        for column in self.reflections.columns:
            new_reflections.add_column(column.label, column.type)

        # Get data
        data_array = np.array(self.reflections, copy=True)
        data = pd.DataFrame(data_array,
                            columns=self.reflections.column_labels(),
                            )
        data.set_index(["H", "K", "L"], inplace=True)

        # add resolutions
        data["res"] = self.reflections.make_d_array() 

        # Truncate by resolution
        data_truncated = data[data["res"] >= resolution.resolution]
        
        # Rem,ove res colum
        data_dropped = data_truncated.drop("res", "columns")
        
        
        # To numpy
        data_dropped_array = data_dropped.to_numpy()
        
        # new data
        new_data = np.hstack([data_dropped.index.to_frame().to_numpy(),
                              data_dropped_array,
                              ]
                             )
        
        # Update
        new_reflections.set_data(new_data)

        # Update resolution
        new_reflections.update_reso()

        return Reflections(new_reflections)
    
    def truncate_reflections(self, index=None) -> Reflections:
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = self.reflections.spacegroup
        new_reflections.set_cell_for_all(self.reflections.cell)

        # Add dataset
        new_reflections.add_dataset("truncated")

        # Add columns
        for column in self.reflections.columns:
            new_reflections.add_column(column.label, column.type)

        # Get data
        data_array = np.array(self.reflections, copy=True)
        data = pd.DataFrame(data_array,
                            columns=self.reflections.column_labels(),
                            )
        data.set_index(["H", "K", "L"], inplace=True)

        # Truncate by index
        data_indexed = data.loc[index]

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

        return Reflections(new_reflections)

    def spacegroup(self):
        return self.reflections.spacegroup

    def columns(self):
        return self.reflections.column_labels()

    def missing(self, structure_factors: StructureFactors, resolution: Resolution) -> pd.DataFrame:
        all_data = np.array(self.reflections, copy=True)
        resolution_array = self.reflections.make_d_array()

        table = pd.DataFrame(data=all_data, columns=self.reflections.column_labels())

        reflections_in_resolution = table[resolution_array >= resolution.to_float()]

        amplitudes = reflections_in_resolution[structure_factors.f]

        missing = reflections_in_resolution[amplitudes == 0]

        return missing

    def common_set(self, other_reflections: Reflections):
        # Index own reflections
        reflections_array = np.array(self.reflections, copy=False, )
        hkl_dict = {}
        f_index = self.reflections.column_labels().index("F")
        for i, row in enumerate(reflections_array):
            hkl = (row[0], row[1], row[2])
            if not np.isnan(row[f_index]):
                hkl_dict[hkl] = i

        # Index the other array
        other_reflections_array = np.array(other_reflections.reflections, copy=False, )
        other_hkl_dict = {}
        f_other_index = other_reflections.reflections.column_labels().index("F")
        for i, row in enumerate(other_reflections_array):
            hkl = (row[0], row[1], row[2])
            if not np.isnan(row[f_other_index]):
                other_hkl_dict[hkl] = i

        # Allocate the masks
        self_mask = np.zeros(reflections_array.shape[0],
                             dtype=np.bool,
                             )

        other_mask = np.zeros(other_reflections_array.shape[0],
                              dtype=np.bool,
                              )

        # Fill the masks
        for hkl, index in hkl_dict.items():
            try:
                other_index = other_hkl_dict[hkl]
                self_mask[index] = True
                other_mask[other_index] = True

            except:
                continue

        return self_mask, other_mask

    # TODO: Make this work reasonably?
    def scale_reflections(self, other: Reflections, cut: float = 99.6):

        data_table = pd.DataFrame(data=np.array(self.reflections),
                                      columns=self.reflections.column_labels(),
                                      index=["H", "K", "L"],
                                      )
        data_other_table = pd.DataFrame(data=np.array(other.reflections),
                                            columns=other.reflections.column_labels(),
                                            index=["H", "K", "L"],
                                            )

        # Set resolutions
        data_table["1_d2"] = self.reflections.make_1_d2_array()
        data_other_table["1_d2"] = other.reflections.make_1_d2_array()

        # Get common indexes
        data_index = data_table[~data_table["F"].isna()].index.to_flat_index()
        data_other_index = data_other_table[~data_other_table["F"].isna()].to_flat_index()
        intersection_index = data_index.intersection(data_other_index)
        intersection_list = intersection_index.to_list()

        # Select common data
        data_common_table = data_table[intersection_list]
        data_other_common_table = data_other_table[intersection_list]

        # Select common amplitudes
        f_array = data_common_table["F"].to_numpy()
        f_other_array = data_other_common_table["F"].to_numpy()

        # Select common resolutions
        res_array = data_common_table["1_d2"].to_numpy()
        res_other_array = data_other_common_table["1_d2"].to_numpy()

        min_scale_list = []
        for i in range(6):

            # Trunate outliers
            diff_array = np.abs(f_array - f_other_array)
            high_diff = np.percentile(diff_array, cut)


            x_truncated = f_array[diff_array < high_diff]
            y_truncated = f_other_array[diff_array < high_diff]

            x_r_truncated = res_array[diff_array < high_diff]
            y_r_truncated = res_other_array[diff_array < high_diff]

            # Interpolate
            knn_y = neighbors.RadiusNeighborsRegressor(0.01)
            knn_y.fit(y_r_truncated.reshape(-1, 1),
                      y_truncated.reshape(-1, 1),
                      )

            knn_x = neighbors.RadiusNeighborsRegressor(0.01)
            knn_x.fit(x_r_truncated.reshape(-1, 1),
                      x_truncated.reshape(-1, 1),
                      )

            sample_grid = np.linspace(min(y_r_truncated), max(y_r_truncated), 100)

            x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)
            y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)


            # optimise scale
            scales = []
            rmsds = []

            for scale in np.linspace(-4, 4, 280):

                x = x_f
                y_s = y_f * np.exp(scale * sample_grid)

                rmsd = np.sum(np.square(x - y_s))

                scales.append(scale)
                rmsds.append(rmsd)

            min_scale = scales[np.argmin(np.log(rmsds))] / -0.5
            min_scale_list.append(min_scale)

            x_all = x_truncated
            y_all = y_truncated

            x_r_all = x_r_truncated
            y_r_all = y_r_truncated
            
    def __getstate__(self):
        return (MtzPython.from_gemmi(self.reflections), self.path)
    
    def __setstate__(self, data: Tuple[MtzPython, Path]):
        reflections_python = data[0]
        path = data[1]
        reflections = reflections_python.to_gemmi()
        self.reflections = reflections
        self.path = path


@dataclasses.dataclass()
class Dataset:
    structure: Structure
    reflections: Reflections
    smoothing_factor: float = 0.0

    @staticmethod
    def from_files(pdb_file: Path, mtz_file: Path):
        strucure: Structure = Structure.from_file(pdb_file)
        reflections: Reflections = Reflections.from_file(mtz_file)

        return Dataset(structure=strucure,
                       reflections=reflections,
                       )

    def truncate_resolution(self, resolution: Resolution) -> Dataset:
        return Dataset(self.structure,
                       self.reflections.truncate_resolution(resolution,
                                                            )
                       )
        
    def truncate_reflections(self, index=None) -> Dataset:
        return Dataset(self.structure,
                        self.reflections.truncate_reflections( index,
                                                            )
                    )

    def scale_reflections(self, reference: Reference):
        new_reflections = self.reflections.scale_reflections(reference)
        return Dataset(self.structure,
                       new_reflections,
                       )
        
    def common_reflections(self, 
                           reference_ref: Reflections,
                           structure_factors: StructureFactors,
                           ):
        # Get own reflections
        dtag_reflections = self.reflections.reflections
        dtag_reflections_array = np.array(dtag_reflections, copy=True)
        dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                            columns=dtag_reflections.column_labels(),
                                            )
        dtag_reflections_table.set_index(["H", "K", "L"], inplace=True)
        dtag_flattened_index = dtag_reflections_table[~dtag_reflections_table[structure_factors.f].isna()].index.to_flat_index()
        
        # Get reference
        reference_reflections = reference_ref.reflections
        reference_reflections_array = np.array(reference_reflections, copy=True)
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                            columns=reference_reflections.column_labels(),
                                            )
        reference_reflections_table.set_index(["H", "K", "L"], inplace=True)
        reference_flattened_index = reference_reflections_table[~reference_reflections_table[structure_factors.f].isna()].index.to_flat_index()
        
        
        running_index = dtag_flattened_index.intersection(reference_flattened_index)

        return running_index.to_list()
        
        
        
    def smooth(self, reference: Reference, structure_factors: StructureFactors):
        reference_dataset = reference.dataset
        
        # Get common set of reflections
        common_reflections = self.common_reflections(reference_dataset.reflections,
                                                        structure_factors,
                                                        )
        
        
        # Truncate 
        truncated_reference = reference.dataset.truncate_reflections(common_reflections)
        truncated_dataset = self.truncate_reflections(common_reflections)
        
        # Refference array 
        reference_reflections = truncated_reference.reflections.reflections
        reference_reflections_array = np.array(reference_reflections,
                                            copy=True,
                                            )
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                                columns=reference_reflections.column_labels(),
                                                )
        reference_f_array = reference_reflections_table[structure_factors.f].to_numpy()
        
        # Dtag array
        dtag_reflections = truncated_dataset.reflections.reflections                           
        dtag_reflections_array = np.array(dtag_reflections,
                                        copy=True,
                                    )
        dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                            columns=dtag_reflections.column_labels(),
                                            )        
        dtag_f_array = dtag_reflections_table[structure_factors.f].to_numpy()

        # Resolution array
        resolution_array = reference_reflections.make_1_d2_array()

        # Prepare optimisation
        x = reference_f_array
        y = dtag_f_array
        
        r = resolution_array
        
        sample_grid = np.linspace(min(r), max(r), 100)
        
        knn_x = neighbors.RadiusNeighborsRegressor(0.01)
        knn_x.fit(r.reshape(-1,1), 
                    x.reshape(-1,1),
                    )
        x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)

        scales = []
        rmsds = []
        
        # Optimise the scale factor
        for scale in np.linspace(-4,4,100):
            y_s = y * np.exp(scale * r)
            knn_y = neighbors.RadiusNeighborsRegressor(0.01)
            knn_y.fit(r.reshape(-1,1), 
                    y_s.reshape(-1,1),
                    )


            y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

            rmsd = np.sum(np.abs(x_f-y_f)) 

            scales.append(scale)
            rmsds.append(rmsd)
            
        min_scale = scales[np.argmin(rmsds)]

        
        # Get the original reflections
        original_reflections = self.reflections.reflections
        
        original_reflections_array = np.array(original_reflections,
                                            copy=True,
                                            )
        
        original_reflections_table = pd.DataFrame(original_reflections_array,
                                        columns=reference_reflections.column_labels(),
                                        )
        
        f_array = original_reflections_table[structure_factors.f]
        
        f_scaled_array = f_array * np.exp(min_scale*original_reflections.make_1_d2_array())
        
        original_reflections_table[structure_factors.f] = f_scaled_array
        
        # New reflections
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = original_reflections.spacegroup
        new_reflections.set_cell_for_all(original_reflections.cell)

        # Add dataset
        new_reflections.add_dataset("scaled")

        # Add columns
        for column in original_reflections.columns:
            new_reflections.add_column(column.label, column.type)
        
        # Update
        new_reflections.set_data(original_reflections_table.to_numpy())

        # Update resolution
        new_reflections.update_reso() 
                
        # Create new dataset
        smoothed_dataset = Dataset(self.structure,
                              Reflections(new_reflections),
                            )    

        
        return smoothed_dataset

    
        


@dataclasses.dataclass()
class Datasets:
    datasets: typing.Dict[Dtag, Dataset]

    @staticmethod
    def from_dir(pandda_fs_model: PanDDAFSModel):
        datasets = {}
        for dtag, dataset_dir in pandda_fs_model.data_dirs.to_dict().items():
            dataset: Dataset = Dataset.from_files(dataset_dir.input_pdb_file,
                                                  dataset_dir.input_mtz_file,
                                                  )

            datasets[dtag] = dataset

        return Datasets(datasets)

    def __getitem__(self, item):
        return self.datasets[item]

    def remove_dissimilar_models(self, reference: Reference, max_rmsd_to_reference: float) -> Datasets:


        new_dtags = filter(lambda dtag: (RMSD.from_structures(self.datasets[dtag].structure,
                                                              reference.dataset.structure,
                                                              )).to_float() < max_rmsd_to_reference,
                           self.datasets,
                           )

        new_datasets = {dtag: self.datasets[dtag] for dtag in new_dtags}

        return Datasets(new_datasets)

    def remove_invalid_structure_factor_datasets(self,
                                                 structure_factors: StructureFactors,
                                                 ) -> Datasets:
        new_dtags = filter(lambda dtag: (structure_factors.f in self.datasets[dtag].reflections.columns()) and (
                structure_factors.phi in self.datasets[dtag].reflections.columns()),
                           self.datasets,
                           )

        new_datasets = {dtag: self.datasets[dtag] for dtag in new_dtags}

        return Datasets(new_datasets)
    
    def trunate_num_datasets(self, num_datasets: int):
        new_datasets = {dtag: self.datasets[dtag] for i, dtag in enumerate(self.datasets) if i < num_datasets}
        return Datasets(new_datasets)

    def remove_incomplete_up_to_resolution_datasets(self, structure_factors, resolution: Resolution):
        no_missing_reflections_dtags = filter(
            lambda dtag: len(self.datasets[dtag].reflections.missing(structure_factors.f,
                                                                     resolution=resolution,
                                                                     )
                             ) > 0,
            self.datasets,
        )

        new_datasets = {dtag: self.datasets[dtag] for dtag in no_missing_reflections_dtags}

        return Datasets(new_datasets)

    def remove_low_resolution_datasets(self, resolution_cutoff):
        high_resolution_dtags = filter(
            lambda dtag: self.datasets[dtag].reflections.resolution().to_float() < resolution_cutoff,
            self.datasets,
        )

        new_datasets = {dtag: self.datasets[dtag] for dtag in high_resolution_dtags}

        return Datasets(new_datasets)

    def scale_reflections(self, reference: Reference):
        # Scale to the reference dataset

        new_datasets = {}
        for dtag in self.datasets:
            dataset = self.datasets[dtag]
            new_dataset = dataset.scale_refelections()
            new_datasets[dtag] = new_dataset

        return Dataset(new_dataset)

    def remove_bad_rfree(self, max_rfree: float):
        good_rfree_dtags = filter(
            lambda dtag: self.datasets[dtag].structure.rfree().to_float() < max_rfree,
            self.datasets,
        )

        new_datasets = {dtag: self.datasets[dtag] for dtag in good_rfree_dtags}

        return Datasets(new_datasets)

    def remove_dissimilar_space_groups(self, reference: Reference):
        same_spacegroup_datasets = filter(
            lambda dtag: self.datasets[dtag].reflections.spacegroup() == reference.dataset.reflections.spacegroup(),
            self.datasets,
        )

        new_datasets = {dtag: self.datasets[dtag] for dtag in same_spacegroup_datasets}

        return Datasets(new_datasets)

    def remove_bad_wilson(self, max_wilson_plot_z_score: float):
        return self

    def common_reflections(self, structure_factors: StructureFactors):
        
        running_index = None
        for dtag in self.datasets:
            dataset = self.datasets[dtag]
            reflections = dataset.reflections.reflections
            reflections_array = np.array(reflections, copy=True)
            reflections_table = pd.DataFrame(reflections_array,
                                             columns=reflections.column_labels(),
                                             )
            reflections_table.set_index(["H", "K", "L"], inplace=True)
            flattened_index = reflections_table[~reflections_table[structure_factors.f].isna()].index.to_flat_index()
            if running_index is None:
                running_index = flattened_index
            running_index = running_index.intersection(flattened_index)


        return running_index.to_list()


    def truncate(self, resolution: Resolution, structure_factors: StructureFactors) -> Datasets:
        new_datasets_resolution = {}
        
        # Truncate by common resolution
        for dtag in self.datasets:
            truncated_dataset = self.datasets[dtag].truncate_resolution(resolution,)

            new_datasets_resolution[dtag] = truncated_dataset
            
        dataset_resolution_truncated = Datasets(new_datasets_resolution)
            
        # Get common set of reflections
        common_reflections = dataset_resolution_truncated.common_reflections(structure_factors)
        
        # truncate on reflections
        new_datasets_reflections = {}
        for dtag in dataset_resolution_truncated:
            truncated_dataset = dataset_resolution_truncated[dtag].truncate_reflections(common_reflections,
                                                             )

            new_datasets_reflections[dtag] = truncated_dataset

        return Datasets(new_datasets_reflections)
    
    
    def smooth_datasets(self, 
                        reference: Reference, 
                        structure_factors: StructureFactors,
                        cut = 97.5,
                        mapper=False,
                        ):
        
        if mapper:
            keys = list(self.datasets.keys())
            
            results = mapper(
                                           delayed(
                                               self[key].smooth)(
                                                   reference,
                                                   structure_factors
                                                   )
                                               for key
                                               in keys
                                       )
                                       
            smoothed_datasets = {keys[i]: results[i]
                for i, key
                in enumerate(keys)
                }
        
        
        else:
            smoothed_datasets = {}
            for dtag in self.datasets:
                dataset = self.datasets[dtag]

                smoothed_dataset = dataset.smooth(reference,
                                                structure_factors,
                                                )
                smoothed_datasets[dtag] = smoothed_dataset
            
        return Datasets(smoothed_datasets)
        
    def smooth(self, 
               reference_reflections: gemmi.Mtz, 
            structure_factors: StructureFactors,
            cut = 97.5,
            ):
    
        reference_reflections_array = np.array(reference_reflections,
                                            copy=True,
                                            )
        
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                                columns=reference_reflections.column_labels(),
                                                )
            
        reference_f_array = reference_reflections_table[structure_factors.f].to_numpy()
        
        resolution_array = reference_reflections.make_1_d2_array()

        new_reflections_dict = {}
        smoothing_factor_dict = {}
        for dtag in self.datasets:

            dtag_reflections = self.datasets[dtag].reflections.reflections
                                        
                                        
            dtag_reflections_array = np.array(dtag_reflections,
                                            copy=True,
                                        )
            dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                                columns=dtag_reflections.column_labels(),
                                                )
            
            dtag_f_array = dtag_reflections_table[structure_factors.f].to_numpy()

            x = reference_f_array
            y = dtag_f_array
            
            r = resolution_array
            
            sample_grid = np.linspace(min(r), max(r), 100)
            
            knn_x = neighbors.RadiusNeighborsRegressor(0.01)
            knn_x.fit(r.reshape(-1,1), 
                      x.reshape(-1,1),
                      )
            x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)

            scales = []
            rmsds = []

            for scale in np.linspace(-4,4,100):
                y_s = y * np.exp(scale * r)
                knn_y = neighbors.RadiusNeighborsRegressor(0.01)
                knn_y.fit(r.reshape(-1,1), 
                        y_s.reshape(-1,1),
                        )


                y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

                rmsd = np.sum(np.abs(x_f-y_f)) 

                scales.append(scale)
                rmsds.append(rmsd)
                
            min_scale = scales[np.argmin(rmsds)]

            
            f_array = dtag_reflections_table[structure_factors.f]
            
            f_scaled_array = f_array * np.exp(min_scale*resolution_array)
            
            dtag_reflections_table[structure_factors.f] = f_scaled_array
            
            # New reflections
            new_reflections = gemmi.Mtz(with_base=False)

            # Set dataset properties
            new_reflections.spacegroup = dtag_reflections.spacegroup
            new_reflections.set_cell_for_all(dtag_reflections.cell)

            # Add dataset
            new_reflections.add_dataset("scaled")

            # Add columns
            for column in dtag_reflections.columns:
                new_reflections.add_column(column.label, column.type)
            
            # Update
            new_reflections.set_data(dtag_reflections_table.to_numpy())

            # Update resolution
            new_reflections.update_reso() 
                    
            new_reflections_dict[dtag] = new_reflections
            smoothing_factor_dict[dtag] = min_scale
            
            
        # Create new dataset
        new_datasets_dict = {}
        for dtag in self.datasets:
            dataset = self.datasets[dtag]
            structure = dataset.structure
            new_reflections = new_reflections_dict[dtag]
            
            smoothing_factor = smoothing_factor_dict[dtag]
            
            new_dataset = Dataset(structure,
                                Reflections(new_reflections),
                                smoothing_factor=smoothing_factor
                                )    
            new_datasets_dict[dtag] = new_dataset
            
            
        
        return Datasets(new_datasets_dict)
    
    # def smooth_dep(self, reference_reflections: Reflections,
    #            structure_factors: StructureFactors,
    #            cut = 97.5,
    #            ):
        
    #     reference_reflections_array = np.array(reference_reflections,
    #                                            copy=False,
    #                                            )
        
    #     reference_reflections_table = pd.DataFrame(reference_reflections_array,
    #                                               columns=reference_reflections.column_labels(),
    #                                               )
            
    #     reference_f_array = reference_reflections_table[structure_factors.f].to_numpy()
        
    #     resolution_array = reference_reflections.make_1_d2_array()

    #     new_reflections_dict = {}
    #     for dtag in self.datasets:

    #         dtag_reflections = self.datasets[dtag].reflections.reflections
                                        
                                        
    #         dtag_reflections_array = np.array(dtag_reflections,
    #                                           copy=False,
    #                                     )
    #         dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
    #                                               columns=dtag_reflections.column_labels(),
    #                                               )
            
    #         dtag_f_array = dtag_reflections_table[structure_factors.f].to_numpy()




    #         min_scale_list = []

    #         selected = np.full(dtag_f_array.shape, 
    #                            True,
    #                         )

    #         for i in range(6):
    #             x = dtag_f_array[selected]
    #             y = reference_f_array[selected]
                
    #             x_r = resolution_array[selected]
    #             y_r = resolution_array[selected]

    #             scales = []
    #             rmsds = []

    #             for scale in np.linspace(-4,4,100):
    #                 y_s = y * np.exp(scale * y_r)
    #                 knn_y = neighbors.RadiusNeighborsRegressor(0.01)
    #                 knn_y.fit(y_r.reshape(-1,1), 
    #                         y_s.reshape(-1,1),
    #                         )

    #                 knn_x = neighbors.RadiusNeighborsRegressor(0.01)
    #                 knn_x.fit(x_r.reshape(-1,1), 
    #                         x.reshape(-1,1),
    #                                                 )

    #                 sample_grid = np.linspace(min(y_r), max(y_r), 100)

    #                 x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)
    #                 y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

    #                 rmsd = np.sum(np.square(x_f-y_f)) 

    #                 scales.append(scale)
    #                 rmsds.append(rmsd)
                    
    #             min_scale = scales[np.argmin(rmsds)]
    #             min_scale_list.append(min_scale)
                
           
    #             selected_copy = selected.copy()
                
    #             diff_array = np.abs(x-(y*np.exp(min_scale*y_r)))

    #             high_diff = np.percentile(diff_array, 
    #                                     cut,
    #                                     )
                
    #             diff_mask = diff_array > high_diff
                
    #             selected_index_tuple = np.nonzero(selected)
    #             high_diff_mask = selected_index_tuple[0][diff_mask]  
                
    #             selected[high_diff_mask] = False
                
    #         min_scale = min_scale_list[-1]
            
            
    #         f_array = dtag_reflections_table[structure_factors.f]
            
    #         f_scaled_array = f_array * np.exp(min_scale*resolution_array)
            
    #         dtag_reflections_table[structure_factors.f] = f_scaled_array
            
    #         # New reflections
    #         new_reflections = gemmi.Mtz(with_base=False)

    #         # Set dataset properties
    #         new_reflections.spacegroup = dtag_reflections.spacegroup
    #         new_reflections.set_cell_for_all(dtag_reflections.cell)

    #         # Add dataset
    #         new_reflections.add_dataset("truncated")

    #         # Add columns
    #         for column in dtag_reflections.columns:
    #             new_reflections.add_column(column.label, column.type)
            
    #         # Update
    #         new_reflections.set_data(dtag_reflections_table.to_numpy())

    #         # Update resolution
    #         new_reflections.update_reso() 
                       
    #         new_reflections_dict[dtag] = new_reflections
            
            
    #     # Create new dataset
    #     new_datasets_dict = {}
    #     for dtag in self.datasets:
    #         dataset = self.datasets[dtag]
    #         structure = dataset.structure
    #         reflections = new_reflections_dict[dtag]
    #         new_dataset = Dataset(structure,
    #                               Reflections(reflections),
    #                               )    
    #         new_datasets_dict[dtag] = new_dataset
            
            
        
    #     return Datasets(new_datasets_dict)



                

    def __iter__(self):
        for dtag in self.datasets:
            yield dtag

    def from_dtags(self, dtags: typing.List[Dtag]):
        new_datasets = {dtag: self.datasets[dtag] for dtag in dtags}
        return Datasets(new_datasets)


@dataclasses.dataclass()
class Reference:
    dtag: Dtag
    dataset: Dataset

    @staticmethod
    def from_datasets(datasets: Datasets):
        resolutions: typing.Dict[Dtag, Resolution] = {}
        for dtag in datasets:
            resolutions[dtag] = datasets[dtag].reflections.resolution()

        min_resolution_dtag = min(resolutions,
                                  key=lambda dtag: resolutions[dtag].to_float(),
                                  )

        min_resolution_structure = datasets[min_resolution_dtag].structure
        min_resolution_reflections = datasets[min_resolution_dtag].reflections

        return Reference(min_resolution_dtag,
                         datasets[min_resolution_dtag]
                         )


@dataclasses.dataclass()
class Partitioning:
    partitioning: typing.Dict[ResidueID, typing.Dict[typing.Tuple[int], typing.Tuple[float]]]
    protein_mask: gemmi.Int8Grid
    symmetry_mask: gemmi.Int8Grid

    def __getitem__(self, item: ResidueID):
        return self.partitioning[item]

    @staticmethod
    def from_reference(reference: Reference,
                       grid: gemmi.FloatGrid,
                       mask_radius: float,
                       mask_radius_symmetry: float,
                       ):

        #
        # array = np.array(grid, copy=False)
        #
        # spacing = np.array([grid.nu, grid.nv, grid.nw])
        #
        # poss = []
        # res_indexes = {}
        # i = 0
        # for model in reference.dataset.structure.structure:
        #     for chain in model:
        #         for res in chain.get_polymer():
        #             ca = res["CA"][0]
        #
        #             position = ca.pos
        #
        #             fractional = grid.unit_cell.fractionalize(position)
        #
        #             poss.append(fractional)
        #
        #             res_indexes[i] = ResidueID.from_residue_chain(model, chain, res)
        #             i = i + 1
        #
        # ca_position_array = np.array([[x for x in pos] for pos in poss])
        #
        # kdtree = spatial.KDTree(ca_position_array)
        #
        # mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        # mask.spacegroup = grid.spacegroup
        # mask.set_unit_cell(grid.unit_cell)
        # for atom in reference.dataset.structure.protein_atoms():
        #     pos = atom.pos
        #     mask.set_points_around(pos,
        #                            radius=mask_radius,
        #                            value=1,
        #                            )
        #
        # mask_array = np.array(mask, copy=False)
        #
        # coord_array = np.argwhere(mask_array == 1)
        #
        # # points_array = PointsArray.from_array(coord_array)
        # #
        # # position_array = PositionsArray.from_points_array(points_array)
        #
        # query_points = coord_array / spacing
        #
        # distances, indexes = kdtree.query(query_points)
        #
        # partitions = {}
        # for i, coord_as_array in enumerate(coord_array):
        #     coord = (coord_as_array[0], coord_as_array[1], coord_as_array[2])
        #
        #     res_num = indexes[i]
        #
        #     res_id = res_indexes[res_num]
        #
        #     if res_id not in partitions:
        #         partitions[res_id] = {}
        #
        #     partitions[res_id][coord] = grid.unit_cell.orthogonalize(gemmi.Fractional(coord[0] / spacing[0],
        #                                                                               coord[1] / spacing[1],
        #                                                                               coord[2] / spacing[2],
        #                                                                               )
        #                                                              )
        #
        # symmetry_mask = Partitioning.get_symmetry_contact_mask(reference.dataset.structure, mask, mask_radius)

        return Partitioning.from_structure(reference.dataset.structure,
                                           grid,
                                           mask_radius,
                                           mask_radius_symmetry,
                                           )

        # return Partitioning(partitions, mask, symmetry_mask)
    
    @staticmethod
    def get_coord_tuple(grid, ca_position_array, structure: Structure, mask_radius: float=6.0, buffer: float=3.0):
            # Get the bounds
            min_x = ca_position_array[:, 0].min() - buffer
            max_x = ca_position_array[:, 0].max() + buffer
            min_y = ca_position_array[:, 1].min() - buffer
            max_y = ca_position_array[:, 1].max() + buffer
            min_z = ca_position_array[:, 2].min() - buffer
            max_z = ca_position_array[:, 2].max() + buffer
            
            # Get the upper and lower bounds of the grid as positions
            grid_min_cart = gemmi.Position(min_x, min_y, min_z)
            grid_max_cart = gemmi.Position(max_x, max_y, max_z)
            
            # Get them as fractions of the unit cell
            grid_min_frac = grid.unit_cell.fractionalize(grid_min_cart)
            grid_max_frac = grid.unit_cell.fractionalize(grid_max_cart)
            
            # Get them as coords
            grid_min_coord = [int(grid_min_frac[0]*grid.nu), int(grid_min_frac[1]*grid.nv), int(grid_min_frac[2]*grid.nw),]
            grid_max_coord = [int(grid_max_frac[0]*grid.nu), int(grid_max_frac[1]*grid.nv), int(grid_max_frac[2]*grid.nw),]
            
            # Get these as fractions
            # fractional_grid_min = [
            #     grid.unit_cell.fractionalize(grid_min_coord[0]), 
            #     grid.unit_cell.fractionalize(grid_min_coord[1]),
            #     grid.unit_cell.fractionalize(grid_min_coord[2]),
            # ]
            fractional_grid_min = [
                grid_min_coord[0] * (1.0 / grid.nu),
                grid_min_coord[1] * (1.0 / grid.nv),
                grid_min_coord[2] * (1.0 / grid.nw),
            ]
            # fractional_grid_max = [
            #     grid.unit_cell.fractionalize(grid_max_coord[0]),
            #     grid.unit_cell.fractionalize(grid_max_coord[1]),
            #     grid.unit_cell.fractionalize(grid_max_coord[2]),
            #     ]
            fractional_grid_max = [
                grid_max_coord[0] * (1.0 / grid.nu),
                grid_max_coord[1] * (1.0 / grid.nv),
                grid_max_coord[2] * (1.0 / grid.nw),
            ]
            fractional_diff = [
                fractional_grid_max[0]-fractional_grid_min[0],
                fractional_grid_max[1]-fractional_grid_min[1],
                fractional_grid_max[2]-fractional_grid_min[2],
            ]

            # Get the grid of points around the protein

            coord_product = itertools.product(
                range(grid_min_coord[0], grid_max_coord[0]),
                range(grid_min_coord[1], grid_max_coord[1]),
                range(grid_min_coord[2], grid_max_coord[2]),
            )
                                        
            coord_array = np.array([[x, y, z] for x, y, z in coord_product])
            
            coord_tuple = (coord_array[:, 0],
                           coord_array[:, 1],
                           coord_array[:, 2],
                           )
            
            # Get the corresponding protein grid
            protein_grid = gemmi.Int8Grid(
                grid_max_coord[0]-grid_min_coord[0],
                grid_max_coord[1]-grid_min_coord[1],
                grid_max_coord[2]-grid_min_coord[2],
                                          )
            protein_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
            protein_grid_unit_cell = gemmi.UnitCell(
                grid.unit_cell.a * fractional_diff[0],
                grid.unit_cell.b * fractional_diff[0],
                grid.unit_cell.c * fractional_diff[0],
                grid.unit_cell.alpha,
                grid.unit_cell.beta,
                grid.unit_cell.gamma,
            )
            protein_grid.set_unit_cell(protein_grid_unit_cell)
            
            # Mask
            for atom in structure.all_atoms():
                pos = atom.pos
                pos_transformed = gemmi.Position(pos.x - grid_min_cart[0],
                                                 pos.y - grid_min_cart[1],
                                                 pos.z - grid_min_cart[2],
                                                 )
                protein_grid.set_points_around(pos_transformed,
                                        radius=mask_radius,
                                        value=1,
                                        )
            
            
            # # Get the corresponding unit cell points
            # coord_unit_cell_tuple = (np.mod(coord_tuple[0], grid.nu),
            #                          np.mod(coord_tuple[1], grid.nv),
            #                          np.mod(coord_tuple[2], grid.nw),
            #                          )
            
            # Get the corresponging protein_grid points 
            coord_mask_grid_tuple = (
                coord_tuple[0]-grid_min_coord[0],
                coord_tuple[1]-grid_min_coord[1],
                coord_tuple[2]-grid_min_coord[2],
                )
            
            # Check which of them are in the mask
            mask_array = np.array(protein_grid, copy=False, dtype=np.int8)
            in_mask_array_int = mask_array[coord_mask_grid_tuple]
            in_mask_array = in_mask_array_int == 1
            
            # Mask those coords in the tuples
            coord_array_in_mask = (
                coord_tuple[0][in_mask_array],
                coord_tuple[1][in_mask_array],
                coord_tuple[2][in_mask_array],
            ) 
            coord_array_unit_cell_in_mask = (
                coord_mask_grid_tuple[0][in_mask_array],
                coord_mask_grid_tuple[1][in_mask_array],
                coord_mask_grid_tuple[2][in_mask_array],
            )
            
            return coord_array_in_mask, coord_array_unit_cell_in_mask
        
    @staticmethod
    def get_position_list(mask, coord_array):
            positions = []
            for u, v, w in coord_array:
                point = mask.get_point(u, v, w)
                position = mask.point_to_position(point)
                positions.append((position[0],
                                position[1],
                                position[2],
                )
                                )
            return positions
    
        
    @staticmethod
    def from_structure(structure: Structure,
                       grid: gemmi.FloatGrid,
                       mask_radius: float,
                       mask_radius_symmetry: float,
                       ):
        poss = []
        res_indexes = {}
        i = 0
        for model in structure.structure:
            for chain in model:
                for res in chain.get_polymer():
                    if res.name.upper() not in RESIDUE_NAMES:
                        continue
                    
                    ca = res["CA"][0]

                    orthogonal = ca.pos
                    # fractional = grid.unit_cell.fractionalize(orthogonal_raw)
                    # wrapped = fractional.wrap_to_unit()
                    # orthogonal = grid.unit_cell.orthogonalize(wrapped)

                    poss.append(orthogonal)

                    res_indexes[i] = ResidueID.from_residue_chain(model, chain, res)
                    i = i + 1

        ca_position_array = np.array([[x for x in pos] for pos in poss])

        kdtree = spatial.KDTree(ca_position_array)

        mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        mask.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        mask.set_unit_cell(grid.unit_cell)
        for atom in structure.protein_atoms():
            pos = atom.pos
            mask.set_points_around(pos,
                                   radius=mask_radius,
                                   value=1,
                                   )
        mask_array = np.array(mask, copy=False)

        symmetry_mask = Partitioning.get_symmetry_contact_mask(structure, grid, mask, mask_radius_symmetry)


        # Get the positions of the protein masked grid points
        # We need the positions in the protein frame
        # coord_array = np.argwhere(mask_array == 1)

        # positions = []
        # for coord in coord_array:
        #     point = mask.get_point(*coord)
        #     position = mask.point_to_position(point)
        #     positions.append((position[0],
        #                       position[1],
        #                       position[2],
        #     )
        #                      )
        # position_array = np.array(positions)
        
        

        coord_tuple, coord_array_unit_cell_in_mask = Partitioning.get_coord_tuple(
            mask,
            ca_position_array,
            structure,
            mask_radius
            )
        
        coord_array = np.concatenate(
            [
                coord_tuple[0].reshape((-1, 1)),
                coord_tuple[1].reshape((-1, 1)),
                coord_tuple[2].reshape((-1, 1)),
            ], 
            axis=1,
            )
        
        
        
        position_list = Partitioning.get_position_list(mask, coord_array)
        position_array = np.array(position_list)

        distances, indexes = kdtree.query(position_array)

        partitions = {}
        for i, coord_as_array in enumerate(coord_array):
            coord = (int(coord_as_array[0]), 
                     int(coord_as_array[1]), 
                     int(coord_as_array[2]),
                     )
            position = position_list[i]

            res_num = indexes[i]

            res_id = res_indexes[res_num]
            if res_id not in partitions:
                partitions[res_id] = {}

            # partitions[res_id][coord] = gemmi.Position(*position)
            
            coord_unit_cell = (int(coord_array_unit_cell_in_mask[0][i]),
                               int(coord_array_unit_cell_in_mask[1][i]),
                               int(coord_array_unit_cell_in_mask[2][i]),
            )
            partitions[res_id][coord] = position



        return Partitioning(partitions, mask, symmetry_mask)

    def coord_tuple(self):
        
        coord_array = self.coord_array()
        coord_tuple = (coord_array[:,0],
                       coord_array[:,1],
                       coord_array[:,2],
                       )
                
        return coord_tuple
    
    def coord_array(self):
        
        coord_list = []
        for res_id in self.partitioning:
            for coord in self.partitioning[res_id]:
                coord_list.append(coord)
                
        return np.array(coord_list)



    @staticmethod
    def get_symmetry_contact_mask(structure: Structure, grid: gemmi.FloatGrid,
                                  protein_mask: gemmi.Int8Grid,
                                  symmetry_mask_radius: float = 3):
        protein_mask_array = np.array(protein_mask, copy=False, dtype=np.int8)

        mask = gemmi.Int8Grid(*protein_mask_array.shape)
        mask.spacegroup = protein_mask.spacegroup
        mask.set_unit_cell(protein_mask.unit_cell)

        # Mask psacegroup summetry related 
        symops = Symops.from_grid(grid)
        for atom in structure.all_atoms():
            position = atom.pos
            fractional_position = mask.unit_cell.fractionalize(position)
            wrapped_position = fractional_position.wrap_to_unit()
            for symmetry_operation in symops.symops[1:]:
               
                symmetry_position = gemmi.Fractional(*symmetry_operation.apply_to_xyz([wrapped_position[0],
                                                                                       wrapped_position[1],
                                                                                       wrapped_position[2],
                                                                                       ]))
                orthogonal_symmetry_position = mask.unit_cell.orthogonalize(symmetry_position)

                mask.set_points_around(orthogonal_symmetry_position,
                                       radius=symmetry_mask_radius,
                                       value=1,
                                       )

        mask_array = np.array(mask, copy=False, dtype=np.int8)

        equal_mask = protein_mask_array == mask_array

        protein_mask_indicies = np.nonzero(protein_mask_array)
        protein_mask_bool = np.full(protein_mask_array.shape, False)
        protein_mask_bool[protein_mask_indicies] = True

        mask_array[~protein_mask_bool] = 0
        
        # ##################################
        # Mask unit cell overlap
        # ################################
        # Make a temp mask to hold overlap
        mask_unit_cell = gemmi.Int8Grid(*protein_mask_array.shape)
        mask_unit_cell.spacegroup = protein_mask.spacegroup
        mask_unit_cell.set_unit_cell(protein_mask.unit_cell)
        mask_unit_cell_array = np.array(mask_unit_cell, copy=False, dtype=np.int8)
        
        
        # Assign atoms to unit cells
        unit_cell_index_dict = {}
        for atom in structure.all_atoms():
            position = atom.pos
            fractional_position = mask_unit_cell.unit_cell.fractionalize(position)
            
            unit_cell_index_tuple = (int(fractional_position.x),
                         int(fractional_position.y),
                         int(fractional_position.z),
                         )
            
            if unit_cell_index_tuple not in unit_cell_index_dict:
                unit_cell_index_dict[unit_cell_index_tuple] = []
            
            unit_cell_index_dict[unit_cell_index_tuple].append(position)
            
        # Create masks of those unit cells
        unit_cell_mask_dict = {}            
        for unit_cell_index, unit_cell_position_list in unit_cell_index_dict.items():
            unit_cell_mask = gemmi.Int8Grid(*protein_mask_array.shape)
            unit_cell_mask.spacegroup = protein_mask.spacegroup
            unit_cell_mask.set_unit_cell(protein_mask.unit_cell)
            
            for position in unit_cell_position_list:
            
                mask_unit_cell.set_points_around_non_overlapping(
                    position,
                    radius=symmetry_mask_radius,
                    value=1,
                            )
            
            unit_cell_mask_dict[unit_cell_index] = unit_cell_mask

        # Overlay those masks
        for unit_cell_index, unit_cell_mask in unit_cell_mask_dict.items():
            unit_cell_mask_array = np.array(unit_cell_mask, copy=False, dtype=np.int8)
            mask_unit_cell_array[unit_cell_mask_array == 1] += 1
            
        # Select the unit cell overlapping points
        mask_unit_cell_array_mask = mask_unit_cell_array > 1
        
        # mask them in the symmetry mask array
        mask_array[mask_unit_cell_array_mask] = 1

        return mask
    
    def save_maps(self, dir: Path, p1: bool=True):
        # Protein mask
        ccp4 = gemmi.Ccp4Mask()
        ccp4.grid = self.protein_mask
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(0, True)
        ccp4.write_ccp4_map(str(dir / PANDDA_PROTEIN_MASK_FILE))
        
        # Symmetry mask
        ccp4 = gemmi.Ccp4Mask()
        ccp4.grid = self.symmetry_mask
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(0, True)
        ccp4.write_ccp4_map(str(dir / PANDDA_SYMMETRY_MASK_FILE))
        
    
    def __getstate__(self):
        # partitioning_python = PartitoningPython.from_gemmi(self.partitioning)
        partitioning_python = self.partitioning
        protein_mask_python = Int8GridPython.from_gemmi(self.protein_mask)
        symmetry_mask_python = Int8GridPython.from_gemmi(self.symmetry_mask)
        return (partitioning_python,
                protein_mask_python,
                symmetry_mask_python,
                )
        
    def __setstate__(self, data):
        partitioning_gemmi = data[0].to_gemmi()
        protein_mask_gemmi = data[1].to_gemmi()
        symmetry_mask_gemmi = data[2].to_gemmi()
        
        self.partitioning = partitioning_gemmi
        self.proteing_mask = protein_mask_gemmi
        self.symmetry_mask = symmetry_mask_gemmi


@dataclasses.dataclass()
class Grid:
    grid: gemmi.FloatGrid
    partitioning: Partitioning

    @staticmethod
    def from_reference(reference: Reference, mask_radius: float, mask_radius_symmetry: float, sample_rate: float = 3.0):
        unit_cell = Grid.unit_cell_from_reference(reference)
        spacing: typing.List[int] = Grid.spacing_from_reference(reference, sample_rate)

        grid = gemmi.FloatGrid(*spacing)
        grid.unit_cell = unit_cell
        grid.spacegroup = reference.dataset.reflections.spacegroup()

        partitioning = Partitioning.from_reference(reference,
                                                   grid,
                                                   mask_radius,
                                                   mask_radius_symmetry)
        

        return Grid(grid, partitioning)

    def new_grid(self):
        spacing = [self.grid.nu, self.grid.nv, self.grid.nw]
        unit_cell = self.grid.unit_cell
        grid = gemmi.FloatGrid(spacing[0], spacing[1], spacing[2])
        grid.set_unit_cell(unit_cell)
        grid.spacegroup = self.grid.spacegroup
        return grid

    @staticmethod
    def spacing_from_reference(reference: Reference, sample_rate: float = 3.0):
        spacing = reference.dataset.reflections.reflections.get_size_for_hkl(sample_rate=sample_rate)
        return spacing

    @staticmethod
    def unit_cell_from_reference(reference: Reference):
        return reference.dataset.reflections.reflections.cell

    def __getitem__(self, item):
        return self.grid[item]

    def volume(self) -> float:
        unit_cell = self.grid.unit_cell
        return unit_cell.volume

    def size(self) -> int:
        grid = self.grid
        return grid.nu * grid.nv * grid.nw

    def shape(self):
        grid = self.grid
        return [grid.nu, grid.nv, grid.nw]
    
    
    def __getstate__(self):
        grid_python = Int8GridPython.from_gemmi(self.grid)
        partitioning_python = self.partitioning.__getstate__()
        
        return (grid_python, partitioning_python)
    
    def __setstate__(self, data):
        self.partitioning = Partitioning(data[1][0],
                                         data[1][1].to_gemmi(),
                                         data[1][2].to_gemmi(),
                                         )
        self.grid = data[0].to_gemmi()

@dataclasses.dataclass()
class Transform:
    transform: gemmi.Transform
    com_reference: np.array
    com_moving: np.array

    def apply_moving_to_reference(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
        transformed_positions = {}
        for index, position in positions.items():
            rotation_frame_position = gemmi.Position(position[0] - self.com_moving[0],
                                                     position[1] - self.com_moving[1],
                                                     position[2] - self.com_moving[2])
            transformed_vector = self.transform.apply(rotation_frame_position)

            transformed_positions[index] = gemmi.Position(transformed_vector[0] + self.com_reference[0],
                                                          transformed_vector[1] + self.com_reference[1],
                                                          transformed_vector[2] + self.com_reference[2])

        return transformed_positions

    def apply_reference_to_moving(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
        inverse_transform = self.transform.inverse()
        transformed_positions = {}
        for index, position in positions.items():
            rotation_frame_position = gemmi.Position(position[0] - self.com_reference[0],
                                                     position[1] - self.com_reference[1],
                                                     position[2] - self.com_reference[2])
            transformed_vector = inverse_transform.apply(rotation_frame_position)

            transformed_positions[index] = gemmi.Position(transformed_vector[0] + self.com_moving[0],
                                                          transformed_vector[1] + self.com_moving[1],
                                                          transformed_vector[2] + self.com_moving[2])

        return transformed_positions

    @staticmethod
    def from_translation_rotation(translation, rotation, com_reference, com_moving):
        transform = gemmi.Transform()
        transform.vec.fromlist(translation.tolist())
        transform.mat.fromlist(rotation.as_matrix().tolist())

        return Transform(transform, com_reference, com_moving)

    @staticmethod
    def from_residues(previous_res, current_res, next_res, previous_ref, current_ref, next_ref):
        previous_ca_pos = previous_res["CA"][0].pos
        current_ca_pos = current_res["CA"][0].pos
        next_ca_pos = next_res["CA"][0].pos

        previous_ref_ca_pos = previous_ref["CA"][0].pos
        current_ref_ca_pos = current_ref["CA"][0].pos
        next_ref_ca_pos = next_ref["CA"][0].pos

        matrix = np.array([
            Transform.pos_to_list(previous_ca_pos),
            Transform.pos_to_list(current_ca_pos),
            Transform.pos_to_list(next_ca_pos),
        ])
        matrix_ref = np.array([
            Transform.pos_to_list(previous_ref_ca_pos),
            Transform.pos_to_list(current_ref_ca_pos),
            Transform.pos_to_list(next_ref_ca_pos),
        ])

        mean = np.mean(matrix, axis=0)
        mean_ref = np.mean(matrix_ref, axis=0)

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref
        com_moving = mean

        return Transform.from_translation_rotation(vec, rotation, com_reference, com_moving)

    @staticmethod
    def pos_to_list(pos: gemmi.Position):
        return [pos[0], pos[1], pos[2]]

    @staticmethod
    def from_start_residues(current_res, next_res, current_ref, next_ref):
        current_ca_pos = current_res["CA"][0].pos
        next_ca_pos = next_res["CA"][0].pos

        current_ref_ca_pos = current_ref["CA"][0].pos
        next_ref_ca_pos = next_ref["CA"][0].pos

        matrix = np.array([
            Transform.pos_to_list(current_ca_pos),
            Transform.pos_to_list(next_ca_pos),
        ])
        matrix_ref = np.array([
            Transform.pos_to_list(current_ref_ca_pos),
            Transform.pos_to_list(next_ref_ca_pos),
        ])

        mean = np.mean(matrix, axis=0)
        mean_ref = np.mean(matrix_ref, axis=0)

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean

        return Transform.from_translation_rotation(vec, rotation, com_reference, com_moving)
    
    @staticmethod
    def from_atoms(dataset_selection,
                        reference_selection,
                        com_dataset,
                        com_reference,
                        ):
        
        
        
        # mean = np.mean(dataset_selection, axis=0)
        # mean_ref = np.mean(reference_selection, axis=0)
        mean = np.array(com_dataset)
        mean_ref = np.array(com_reference)
        

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = dataset_selection - mean
        de_meaned_ref = reference_selection - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean

        return Transform.from_translation_rotation(vec, rotation, com_reference, com_moving)
        

    @staticmethod
    def from_finish_residues(previous_res, current_res, previous_ref, current_ref):
        previous_ca_pos = previous_res["CA"][0].pos
        current_ca_pos = current_res["CA"][0].pos

        previous_ref_ca_pos = previous_ref["CA"][0].pos
        current_ref_ca_pos = current_ref["CA"][0].pos

        matrix = np.array([
            Transform.pos_to_list(previous_ca_pos),
            Transform.pos_to_list(current_ca_pos),
        ])
        matrix_ref = np.array([
            Transform.pos_to_list(previous_ref_ca_pos),
            Transform.pos_to_list(current_ref_ca_pos),
        ])

        mean = np.mean(matrix, axis=0)
        mean_ref = np.mean(matrix_ref, axis=0)

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean

        return Transform.from_translation_rotation(vec, rotation, com_reference, com_moving)
    
    def __getstate__(self):
        transform_python = TransformPython.from_gemmi(self.transform)
        com_reference = self.com_reference
        com_moving = self.com_moving
        return (transform_python, com_reference, com_moving)
    
    def __setstate__(self, data):
        transform_gemmi = data[0].to_gemmi()
        self.transform = transform_gemmi
        self.com_reference = data[1]
        self.com_moving = data[2]



@dataclasses.dataclass()
class Alignment:
    transforms: typing.Dict[ResidueID, Transform]

    def __getitem__(self, item: ResidueID):
        return self.transforms[item]


    @staticmethod
    def from_dataset(reference: Reference, dataset: Dataset):
        
        dataset_pos_list = []
        reference_pos_list = []

        for model in reference.dataset.structure.structure:
            for chain in model:
                for ref_res in chain.get_polymer():
                    if  ref_res.name.upper() not in RESIDUE_NAMES:
                        continue
                    res_id = ResidueID.from_residue_chain(model, chain, ref_res)
                    
                    dataset_res = dataset.structure[res_id][0]
                    
                    for atom_ref, atom_dataset in zip(ref_res, dataset_res):
                        dataset_pos_list.append([atom_dataset.pos.x, atom_dataset.pos.y, atom_dataset.pos.z, ])
                        reference_pos_list.append([atom_ref.pos.x, atom_ref.pos.y, atom_ref.pos.z, ])
        
        # dataset atom coord matrix
        # dataset_atoms = dataset.structure.protein_atoms()
        # dataset_pos_list = []
        # for atom in dataset_atoms:
        #     dataset_pos_list.append([atom.pos.x, atom.pos.y, atom.pos.z, ])
        dataset_atom_array = np.array(dataset_pos_list)
        
        # Other atom coord matrix
        # reference_atoms = reference.dataset.structure.protein_atoms()
        # reference_pos_list = []
        # for atom in reference_atoms:
        #     reference_pos_list.append([atom.pos.x, atom.pos.y, atom.pos.z, ])
        reference_atom_array = np.array(reference_pos_list)
        
        # dataset kdtree
        dataset_tree = spatial.KDTree(dataset_atom_array)
        # Other kdtree
        # reference_tree = spatial.KDTree(reference_atom_array)
        
        if reference_atom_array.size != dataset_atom_array.size:
            raise AlignmentUnmatchedAtomsError(reference_atom_array,
                                               dataset_atom_array,
                                               )

        transforms = {}

        for model in reference.dataset.structure.structure:
            for chain in model:
                for ref_res in chain.get_polymer():
                    if ref_res.name.upper() not in RESIDUE_NAMES:
                        continue
                    
                    # Get ca pos
                    current_res_id = ResidueID.from_residue_chain(model, chain, ref_res)
                    reference_ca_pos = ref_res["CA"][0].pos
                    
                    #
                    dataset_res = dataset.structure[current_res_id][0]
                    dataset_ca_pos = dataset_res["CA"][0].pos
                    
                    # dataset selection
                    dataset_indexes = dataset_tree.query_ball_point([dataset_ca_pos.x, dataset_ca_pos.y, dataset_ca_pos.z], 
                                                                    7.0,
                                                                    )
                    dataset_selection = dataset_atom_array[dataset_indexes]
                    
                    # other selection
                    # reference_indexes = dataset_tree.query_ball_point([reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z], 
                    #                                                 7.0,
                    #                                                 )
                    reference_selection = reference_atom_array[dataset_indexes]
                    
                    transforms[current_res_id] = Transform.from_atoms(
                        dataset_selection,
                        reference_selection,
                        com_dataset=[dataset_ca_pos.x, dataset_ca_pos.y, dataset_ca_pos.z],
                        com_reference=[reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z],
                    )

        return Alignment(transforms)

    # @staticmethod
    # def from_dataset(reference: Reference, dataset: Dataset):

    #     transforms = {}

    #     for model in dataset.structure.structure:
    #         for chain in model:
    #             for res in chain.get_polymer():
    #                 prev_res = chain.previous_residue(res)
    #                 next_res = chain.next_residue(res)

    #                 if prev_res:
    #                     prev_res_id = ResidueID.from_residue_chain(model, chain, prev_res)
    #                 current_res_id = ResidueID.from_residue_chain(model, chain, res)
    #                 if next_res:
    #                     next_res_id = ResidueID.from_residue_chain(model, chain, next_res)

    #                 if prev_res:
    #                     prev_res_ref = reference.dataset.structure[prev_res_id][0]
    #                 current_res_ref = reference.dataset.structure[current_res_id][0]
    #                 if next_res:
    #                     next_res_ref = reference.dataset.structure[next_res_id][0]

    #                 if not prev_res:
    #                     transform = Transform.from_start_residues(res, next_res,
    #                                                               current_res_ref, next_res_ref)

    #                 if not next_res:
    #                     transform = Transform.from_finish_residues(prev_res, res,
    #                                                                prev_res_ref, current_res_ref)

    #                 if prev_res and next_res:
    #                     transform = Transform.from_residues(prev_res, res, next_res,
    #                                                         prev_res_ref, current_res_ref, next_res_ref,
    #                                                         )

    #                 transforms[current_res_id] = transform

    #             for res in chain.get_polymer():
    #                 prev_res = chain.previous_residue(res)
    #                 next_res = chain.next_residue(res)

    #                 if prev_res:
    #                     prev_res_id = ResidueID.from_residue_chain(model, chain, prev_res)
    #                 current_res_id = ResidueID.from_residue_chain(model, chain, res)
    #                 if next_res:
    #                     next_res_id = ResidueID.from_residue_chain(model, chain, next_res)

    #                 if not prev_res:
    #                     transforms[current_res_id].transform.mat.fromlist(
    #                         transforms[next_res_id].transform.mat.tolist())

    #                 if not next_res:
    #                     transforms[current_res_id].transform.mat.fromlist(
    #                         transforms[prev_res_id].transform.mat.tolist())

    #     return Alignment(transforms)

    def __iter__(self):
        for res_id in self.transforms:
            yield res_id
            
    # def __getstate__(self):
    #     alignment = AlignmentPython.from_gemmi(self)
    #     return alignment
    
    # def __setstate__(self, alignment_python: AlignmentPython):
    #     alignment_gemmi = alignment_python.to_gemmi()
    #     self.transforms = alignment_gemmi


@dataclasses.dataclass()
class Alignments:
    alignments: typing.Dict[Dtag, Alignment]

    @staticmethod
    def from_datasets(reference: Reference, datasets: Datasets):
        alignments = {dtag: Alignment.from_dataset(reference, datasets[dtag])
                      for dtag
                      in datasets
                      }
        return Alignments(alignments)

    def __getitem__(self, item):
        return self.alignments[item]
    
    def __iter__(self):
        for dtag in self.alignments:
            yield dtag
    
    def __getstate__(self):
        
        alignments_python = {}
        for dtag, alignment in self.alignments.items():
            alignment = AlignmentPython.from_gemmi(self)
            alignments_python[dtag] = alignment
        return alignments_python
    
    def __setstate__(self, alignment_python: AlignmentPython):
        alignment_gemmi = alignment_python.to_gemmi()
        self.transforms = alignment_gemmi
    
    
    
    


@dataclasses.dataclass()
class Resolution:
    resolution: float

    @staticmethod
    def from_float(res: float):
        return Resolution(res)

    def to_float(self) -> float:
        return self.resolution


@dataclasses.dataclass()
class Shell:
    number: int
    test_dtags: typing.List[Dtag]
    train_dtags: typing.List[Dtag]
    all_dtags: typing.List[Dtag]
    datasets: Datasets
    res_max: Resolution
    res_min: Resolution


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

        train_dtags = sorted_dtags[:min_characterisation_datasets]

        shells = {}
        shell_num = 0
        shell_dtags = []
        shell_res = datasets[sorted_dtags[-1]].reflections.resolution().resolution
        for dtag in sorted_dtags:
            res = datasets[dtag].reflections.resolution().resolution

            if (len(shell_dtags) >= max_shell_datasets) or (
                    res - shell_res >= high_res_increment):
                all_dtags = list(set(shell_dtags).union(set(train_dtags)))
                shell = Shell(shell_num, 
                              shell_dtags,
                              train_dtags,
                              all_dtags,
                              Datasets({dtag: datasets[dtag] for dtag in datasets
                                        if dtag in shell_dtags or train_dtags}),
                              res_max=Resolution.from_float(shell_res),
                              res_min=Resolution.from_float(res),
                              )
                shells[shell_num] = shell

                shell_dtags = []
                shell_res = res
                shell_num = shell_num + 1

            shell_dtags.append(dtag)

        return Shells(shells)

    def __iter__(self):
        for shell_num in self.shells:
            yield self.shells[shell_num]


@dataclasses.dataclass()
class Xmap:
    xmap: gemmi.FloatGrid

    @staticmethod
    def from_reflections(reflections: Reflections):
        pass

    @staticmethod
    def from_unaligned_dataset(dataset: Dataset, alignment: Alignment, grid: Grid, structure_factors: StructureFactors,
                               sample_rate: float = 3.0):
        
        unaligned_xmap: gemmi.FloatGrid = dataset.reflections.reflections.transform_f_phi_to_map(structure_factors.f,
                                                                                                 structure_factors.phi,
                                                                                                 sample_rate=sample_rate,
                                                                                                 )
        unaligned_xmap_array = np.array(unaligned_xmap, copy=False)
        std = np.std(unaligned_xmap_array)

        # unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

        interpolated_values_tuple = ([], [], [], [])

        for residue_id in alignment:
            alignment_positions: typing.Dict[typing.Tuple[int], gemmi.Position] = grid.partitioning[residue_id]

            transformed_positions: typing.Dict[typing.Tuple[int],
                                               gemmi.Position] = alignment[residue_id].apply_reference_to_moving(
                alignment_positions)

            transformed_positions_fractional: typing.Dict[typing.Tuple[int], gemmi.Fractional] = {
                point: unaligned_xmap.unit_cell.fractionalize(pos) for point, pos in transformed_positions.items()}

            interpolated_values: typing.Dict[typing.Tuple[int],
                                             float] = Xmap.interpolate_grid(unaligned_xmap,
                                                                            transformed_positions_fractional)

            interpolated_values_tuple = (interpolated_values_tuple[0] + [index[0] for index in interpolated_values],
                                         interpolated_values_tuple[1] + [index[1] for index in interpolated_values],
                                         interpolated_values_tuple[2] + [index[2] for index in interpolated_values],
                                         interpolated_values_tuple[3] + [interpolated_values[index] for index in
                                                                         interpolated_values],
                                         )

        new_grid = grid.new_grid()

        grid_array = np.array(new_grid, copy=False)

        grid_array[interpolated_values_tuple[0:3]] = interpolated_values_tuple[3]

        return Xmap(new_grid)
    
    @staticmethod
    def from_unaligned_dataset_c(dataset: Dataset, 
                                 alignment: Alignment, 
                                 grid: Grid, 
                                 structure_factors: StructureFactors,
                                 sample_rate: float = 3.0,
                                 ):
                
                
        unaligned_xmap: gemmi.FloatGrid = dataset.reflections.reflections.transform_f_phi_to_map(structure_factors.f,
                                                                                                 structure_factors.phi,
                                                                                                 sample_rate=sample_rate,
                                                                                                 )
        unaligned_xmap_array = np.array(unaligned_xmap, copy=False)


        std = np.std(unaligned_xmap_array)
        unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std
        
        new_grid = grid.new_grid()
        # Unpack the points, poitions and transforms
        point_list: List[Tuple[int, int, int]] = []
        position_list: List[Tuple[float, float, float]] = []
        transform_list: List[gemmi.transform] = []
        com_moving_list: List[np.array] = []
        com_reference_list: List[np.array] = []
                
        for residue_id, point_position_dict in grid.partitioning.partitioning.items():
            
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
                
                    

        # Interpolate values
        interpolated_grid = gemmi.interpolate_points(unaligned_xmap,
                                 new_grid,
                                 point_list,
                                 position_list,
                                 transform_list,
                                 com_moving_list,
                                 com_reference_list,
                                 )


        return Xmap(interpolated_grid)
    
    def new_grid(self):
        spacing = [self.xmap.nu, self.xmap.nv, self.xmap.nw]
        unit_cell = self.xmap.unit_cell
        grid = gemmi.FloatGrid(spacing[0], spacing[1], spacing[2])
        grid.unit_cell = unit_cell
        grid.spacegroup = self.xmap.spacegroup
        return grid
    
    def resample(
        self,
        xmap: Xmap,
        transform: Transform,  # tranfrom FROM the frame of xmap TO the frame of self 
        sample_rate: float = 3.0,
        ):
        
        unaligned_xmap: gemmi.FloatGrid = self.xmap

        unaligned_xmap_array = np.array(unaligned_xmap, copy=False)
        std = np.std(unaligned_xmap_array)

        unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

        # Copy data into new grid
        new_grid = xmap.new_grid()
        
        # points
        original_point_list = list(
            itertools.product(
                range(new_grid.nu),
                range(new_grid.nv),
                range(new_grid.nw),
            )
        )
        

        # Unpack the points, poitions and transforms
        point_list: List[Tuple[int, int, int]] = []
        position_list: List[Tuple[float, float, float]] = []
        transform_list: List[gemmi.transform] = []
        com_moving_list: List[np.array] = []
        com_reference_list: List[np.array] = []

        transform = transform.transform.inverse()
        com_moving = [0.0,0.0,0.0]
        com_reference = [0.0,0.0,0.0]
        position = [0.0, 0.0, 0.0]
                
        for point in original_point_list:
                        
            point_list.append(point)
            position_list.append(position)
            transform_list.append(transform)
            com_moving_list.append(com_moving)
            com_reference_list.append(com_reference)
                
                    

        # Interpolate values
        interpolated_grid = gemmi.interpolate_points(unaligned_xmap,
                                 new_grid,
                                 point_list,
                                 position_list,
                                 transform_list,
                                 com_moving_list,
                                 com_reference_list,
                                 )


        return Xmap(new_grid)

    @staticmethod
    def from_aligned_map(event_map_reference_grid: gemmi.FloatGrid,
                         dataset: Dataset, alignment: Alignment, grid: Grid,
                         structure_factors: StructureFactors, mask_radius: float,
                         mask_radius_symmetry: float):

        partitioning = Partitioning.from_structure(dataset.structure,
                                                   event_map_reference_grid,
                                                   mask_radius,
                                                   mask_radius_symmetry)


        interpolated_values_tuple = ([], [], [], [])

        for residue_id in alignment:
            alignment_positions: typing.Dict[typing.Tuple[int], gemmi.Position] = partitioning[residue_id]

            transformed_positions: typing.Dict[typing.Tuple[int],
                                               gemmi.Position] = alignment[residue_id].apply_moving_to_reference(
                alignment_positions)

            transformed_positions_fractional: typing.Dict[typing.Tuple[int], gemmi.Fractional] = {
                point: event_map_reference_grid.unit_cell.fractionalize(pos) for point, pos in
                transformed_positions.items()}

            interpolated_values: typing.Dict[typing.Tuple[int],
                                             float] = Xmap.interpolate_grid(event_map_reference_grid,
                                                                            transformed_positions_fractional,
                                                                            )

            interpolated_values_tuple = (interpolated_values_tuple[0] + [index[0] for index in interpolated_values],
                                         interpolated_values_tuple[1] + [index[1] for index in interpolated_values],
                                         interpolated_values_tuple[2] + [index[2] for index in interpolated_values],
                                         interpolated_values_tuple[3] + [interpolated_values[index] for index in
                                                                         interpolated_values],
                                         )

        new_grid = gemmi.FloatGrid(*[event_map_reference_grid.nu,
                                     event_map_reference_grid.nv,
                                     event_map_reference_grid.nw])
        new_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        new_grid.set_unit_cell(event_map_reference_grid.unit_cell)

        grid_array = np.array(new_grid, copy=False)

        grid_array[interpolated_values_tuple[0:3]] = interpolated_values_tuple[3]

        return new_grid
    
    @staticmethod
    def from_aligned_map_c(event_map_reference_grid: gemmi.FloatGrid,
                         dataset: Dataset, alignment: Alignment, grid: Grid,
                         structure_factors: StructureFactors, mask_radius: float,
                         mask_radius_symmetry: float):

        partitioning = Partitioning.from_structure(dataset.structure,
                                                   event_map_reference_grid,
                                                   mask_radius,
                                                   mask_radius_symmetry)
        
        


        new_grid = gemmi.FloatGrid(*[event_map_reference_grid.nu,
                                     event_map_reference_grid.nv,
                                     event_map_reference_grid.nw])
        new_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        new_grid.set_unit_cell(event_map_reference_grid.unit_cell)
        
        # Unpack the points, poitions and transforms
        point_list = []
        position_list = []
        transform_list = []
        com_moving_list = []
        com_reference_list = []
        for residue_id in grid.partitioning.partitioning:
            
            if residue_id in partitioning.partitioning:
                al = alignment[residue_id]
                transform = al.transform
                com_moving = al.com_reference
                com_reference = al.com_moving
                point_position_dict = partitioning[residue_id]
                
                for point, position in point_position_dict.items():
                    point_list.append(point)
                    position_list.append(position)
                    transform_list.append(transform)
                    com_moving_list.append(com_moving)
                    com_reference_list.append(com_reference)
            else:
                continue


        # Interpolate values
        interpolated_grid = gemmi.interpolate_points(
            event_map_reference_grid,
            new_grid,
            point_list,
            position_list,
            transform_list,
            com_moving_list,
            com_reference_list,
            )

        return Xmap(interpolated_grid)


    @staticmethod
    def interpolate_grid(grid: gemmi.FloatGrid,
                         positions: typing.Dict[typing.Tuple[int],
                                                gemmi.Position]) -> typing.Dict[typing.Tuple[int], float]:
        return {coord: grid.interpolate_value(pos) for coord, pos in positions.items()}

    def to_array(self, copy=True):
        return np.array(self.xmap, copy=copy)

    def save(self, path: Path, p1: bool=True):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = self.xmap
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(path))
        
    def __getstate__(self):
        return XmapPython.from_gemmi(self.xmap)
    
    def __setstate__(self, xmap_python: XmapPython):
        self.xmap = xmap_python.to_gemmi()

@dataclasses.dataclass()
class Xmaps:
    xmaps: typing.Dict[Dtag, Xmap]

    @staticmethod
    def from_datasets(datasets: Datasets):
        pass

    @staticmethod
    def from_aligned_datasets(datasets: Datasets, alignments: Alignments, grid: Grid,
                              structure_factors: StructureFactors, sample_rate=3.0,
                              mapper=True,
                              ):
        
        if mapper:
            keys = list(datasets.datasets.keys())
            
            results = mapper(
                delayed(Xmap.from_unaligned_dataset)(
                    datasets[key],
                    alignments[key],
                    grid,
                    structure_factors,
                    sample_rate,
                    )
                for key
                in keys
                )
                                       
            xmaps = {keys[i]: results[i]
                for i, key
                in enumerate(keys)
                }
            
        else:
            
            xmaps = {}
            for dtag in datasets:
                xmap = Xmap.from_unaligned_dataset(datasets[dtag],
                                                alignments[dtag],
                                                grid,
                                                structure_factors,
                                                sample_rate)

                xmaps[dtag] = xmap

        return Xmaps(xmaps)
    
    @staticmethod
    def from_aligned_datasets_c(datasets: Datasets, alignments: Alignments, grid: Grid,
                              structure_factors: StructureFactors, sample_rate=3.0,
                              mapper=False,
                              ):
        
        if mapper:
            
            keys = list(datasets.datasets.keys())
            
            results = mapper(
                                           delayed(Xmap.from_unaligned_dataset_c)(
                                               datasets[key],
                                               alignments[key],
                                               grid,
                                               structure_factors,
                                               sample_rate,
                                               )
                                           for key
                                           in keys
                                       )
                                       
            xmaps = {keys[i]: results[i]
                for i, key
                in enumerate(keys)
                }
            
        else:
            
            xmaps = {}
            for dtag in datasets:
                xmap = Xmap.from_unaligned_dataset_c(
                    datasets[dtag],
                                               alignments[dtag],
                                               grid,
                                               structure_factors,
                                               sample_rate,
                                               )

                xmaps[dtag] = xmap
                
        return Xmaps(xmaps)

    def from_dtags(self, dtags: typing.List[Dtag]):
        new_xmaps = {dtag: self.xmaps[dtag] for dtag in dtags}
        return Xmaps(new_xmaps)

    def __len__(self):
        return len(self.xmaps)

    def __getitem__(self, item):
        return self.xmaps[item]

    def __iter__(self):
        for dtag in self.xmaps:
            yield dtag


@dataclasses.dataclass()
class XmapArray:
    dtag_list: typing.List[Dtag]
    xmap_array: np.ndarray

    def __iter__(self):
        for dtag in self.dtag_list:
            yield dtag

    def __getitem__(self, item):
        index = self.dtag_list.index(item)
        return self.xmap_array[index, :]

    @staticmethod
    def from_xmaps(xmaps: Xmaps,
                   grid: Grid,
                   ):

        mask = grid.partitioning.protein_mask
        mask_array = np.array(mask, copy=False, dtype=np.int8)

        arrays = {}
        for dtag in xmaps:
            xmap = xmaps[dtag]
            xmap_array = xmap.to_array()

            arrays[dtag] = xmap_array[np.nonzero(mask_array)]

        dtag_list = list(arrays.keys())
        xmap_array = np.stack(list(arrays.values()), axis=0)

        return XmapArray(dtag_list, xmap_array)

    def from_dtags(self, dtags: typing.List[Dtag]):
        bool_mask = []
        for dtag in self.dtag_list:
            if dtag in dtags:
                bool_mask.append(True)
            else:
                bool_mask.append(False)

        mask_array = np.array(bool_mask)

        view = self.xmap_array[mask_array]

        return XmapArray(dtags, view)


@dataclasses.dataclass()
class Model:
    mean: np.array
    sigma_is: typing.Dict[Dtag, float]
    sigma_s_m: np.ndarray

    @staticmethod
    def mean_from_xmap_array(masked_train_xmap_array: XmapArray):
        mean_flat = np.mean(masked_train_xmap_array.xmap_array, axis=0)

        return mean_flat

    @staticmethod
    def sigma_is_from_xmap_array(masked_train_xmap_array: XmapArray,
                                 mean_array: np.ndarray,
                                 cut: float,
                                 ):
        # Estimate the dataset residual variability
        sigma_is = {}
        for dtag in masked_train_xmap_array:
            sigma_i = Model.calculate_sigma_i(mean_array,
                                              masked_train_xmap_array[dtag],
                                              cut,
                                              )
            sigma_is[dtag] = sigma_i

        return sigma_is

    @staticmethod
    def sigma_sms_from_xmaps(masked_train_xmap_array: XmapArray,
                             mean_array: np.ndarray,
                             sigma_is: typing.Dict[Dtag, float],
                             ):
        # Estimate the adjusted pointwise variance
        sigma_is_array = np.array([sigma_is[dtag] for dtag in masked_train_xmap_array],
                                  dtype=np.float32)[:, np.newaxis]
        sigma_s_m_flat = Model.calculate_sigma_s_m(mean_array,
                                                   masked_train_xmap_array.xmap_array,
                                                   sigma_is_array,
                                                   )

        return sigma_s_m_flat

    @staticmethod
    def from_mean_is_sms(mean_flat,
                         sigma_is,
                         sigma_s_m_flat,
                         grid: Grid, ):

        mask = grid.partitioning.protein_mask
        mask_array = np.array(mask, copy=False, dtype=np.int8)

        mean = np.zeros(mask_array.shape, dtype=np.float32)
        mean[np.nonzero(mask_array)] = mean_flat

        sigma_s_m = np.zeros(mask_array.shape, dtype=np.float32)
        sigma_s_m[np.nonzero(mask_array)] = sigma_s_m_flat

        return Model(mean,
                     sigma_is,
                     sigma_s_m,
                     )

    @staticmethod
    def from_xmaps(xmaps: Xmaps, grid: Grid, cut: float):
        mask = grid.partitioning.protein_mask
        mask_array = np.array(mask, copy=False, dtype=np.int8)

        arrays = {}
        for dtag in xmaps:
            xmap = xmaps[dtag]
            xmap_array = xmap.to_array()

            arrays[dtag] = xmap_array[np.nonzero(mask_array)]

        stacked_arrays = np.stack(list(arrays.values()), axis=0)
        mean_flat = np.mean(stacked_arrays, axis=0)

        # Estimate the dataset residual variability
        sigma_is = {}
        for dtag in xmaps:
            sigma_i = Model.calculate_sigma_i(mean_flat,
                                              arrays[dtag],
                                              cut)
            sigma_is[dtag] = sigma_i

        # Estimate the adjusted pointwise variance
        sigma_is_array = np.array(list(sigma_is.values()), dtype=np.float32)[:, np.newaxis]
        sigma_s_m_flat = Model.calculate_sigma_s_m(mean_flat,
                                                   stacked_arrays[:60],
                                                   sigma_is_array[:60],
                                                   )

        mean = np.zeros(mask_array.shape, dtype=np.float32)
        mean[np.nonzero(mask_array)] = mean_flat

        sigma_s_m = np.zeros(mask_array.shape, dtype=np.float32)
        sigma_s_m[np.nonzero(mask_array)] = sigma_s_m_flat

        return Model(mean,
                     sigma_is,
                     sigma_s_m,
                     )

    @staticmethod
    def calculate_sigma_i(mean: np.array, array: np.array, cut: float):
        # TODO: Make sure this is actually equivilent
        # Calculated from slope of array - mean distribution against normal(0,1)
        residual = np.subtract(array, mean)
        observed_quantile_estimates = np.sort(residual)
        # sigma_i = np.std(residual)
        # sigma_i = 0.01

        percentiles = np.linspace(0, 1, array.size + 2)[1:-1]
        normal_quantiles = stats.norm.ppf(percentiles)

        below_min_mask = normal_quantiles < (-1.0 * cut)
        above_max_mask = normal_quantiles > cut
        centre_mask = np.full(below_min_mask.shape, True)
        centre_mask[below_min_mask] = False
        centre_mask[above_max_mask] = False

        central_theoretical_quantiles = normal_quantiles[centre_mask]
        central_observed_quantiles = observed_quantile_estimates[centre_mask]

        map_unc, map_off = np.polyfit(x=central_theoretical_quantiles, y=central_observed_quantiles, deg=1)

        return map_unc

    @staticmethod
    def calculate_sigma_s_m(mean: np.array, arrays: np.array, sigma_is_array: np.array):
        # Maximise liklihood of data at m under normal(mu_m, sigma_i + sigma_s_m) by optimising sigma_s_m
        # mean[m]
        # arrays[n,m]
        # sigma_i_array[n]
        #


        func = lambda est_sigma: Model.log_liklihood(est_sigma, mean, arrays, sigma_is_array)

        shape = mean.shape
        num = len(sigma_is_array)

        sigma_ms = Model.vectorised_optimisation_bisect(func,
                                                        0,
                                                        20,
                                                        31,
                                                        arrays.shape
                                                        )

        # sigma_ms = Model.maximise_over_range(func,
        #                                      0,
        #                                      6,
        #                                      150,
        #                                      arrays.shape)

        return sigma_ms

    @staticmethod
    def maximise_over_range(func, start, stop, num, shape):
        xs = np.linspace(start, stop, num)

        x_opt = np.ones(shape[1], dtype=np.float32) * xs[0]  # n
        x_current = np.ones(shape[1], dtype=np.float32) * xs[0]  # n

        y_max = func(x_current[np.newaxis, :])  # n -> 1,n -> n

        for x in xs[1:]:
            x_current[:] = x  # n
            y = func(x_current[np.newaxis, :])  # n -> 1,n -> n
            y_above_y_max_mask = y > y_max  # n
            # y_max[y_above_y_max_mask] = y[y_above_y_max_mask]
            x_opt[y_above_y_max_mask] = x

        return x_opt

    @staticmethod
    def vectorised_optimisation_bf(func, start, stop, num, shape):
        xs = np.linspace(start, stop, num)

        val = np.ones(shape, dtype=np.float32) * xs[0] + 1.0 / 10000000000000000000000.0
        res = np.ones((shape[1], shape[2], shape[3]), dtype=np.float32) * xs[0]

        y_max = func(val)

        for x in xs[1:]:
            val[:, :, :, :] = 1
            val = val * x

            y = func(x)

            y_above_y_max_mask = y > y_max
            y_max[y_above_y_max_mask] = y[y_above_y_max_mask]
            res[y_above_y_max_mask] = x

        return res

    @staticmethod
    def vectorised_optimisation_bisect(func, start, stop, num, shape):
        # Define step 0
        x_lower_orig = (np.ones(shape[1], dtype=np.float32) * start)

        x_upper_orig = np.ones(shape[1], dtype=np.float32) * stop

        f_lower = func(x_lower_orig[np.newaxis, :])

        f_upper = func(x_upper_orig[np.newaxis, :])

        test_mat = f_lower * f_upper

        test_mat_mask = test_mat > 0

        # mask = np.tile(test_mat_mask, (x_lower_orig.shape[0], 1))

        x_lower = x_lower_orig
        x_upper = x_upper_orig

        for i in range(num):
            x_bisect = x_lower + ((x_upper - x_lower) / 2)

            f_lower = func(x_lower[np.newaxis, :])
            f_bisect = func(x_bisect[np.newaxis, :])

            f_lower_positive = f_lower >= 0
            f_lower_negative = f_lower < 0

            f_bisect_positive = f_bisect >= 0
            f_bisect_negative = f_bisect < 0

            f_lower_positive_bisect_negative = f_lower_positive * f_bisect_negative
            f_lower_negative_bisect_positive = f_lower_negative * f_bisect_positive

            f_lower_positive_bisect_positive = f_lower_positive * f_bisect_positive
            f_lower_negative_bisect_negative = f_lower_negative * f_bisect_negative

            x_upper[f_lower_positive_bisect_negative] = x_bisect[f_lower_positive_bisect_negative]
            x_upper[f_lower_negative_bisect_positive] = x_bisect[f_lower_negative_bisect_positive]

            x_lower[f_lower_positive_bisect_positive] = x_bisect[f_lower_positive_bisect_positive]
            x_lower[f_lower_negative_bisect_negative] = x_bisect[f_lower_negative_bisect_negative]

        # Replace original vals
        x_lower[test_mat_mask] = 0.0

        return x_lower

    @staticmethod
    def log_liklihood(est_sigma, est_mu, obs_vals, obs_error):
        """Calculate the value of the differentiated log likelihood for the values of mu, sigma"""

        term1 = np.square(obs_vals - est_mu) / np.square(np.square(est_sigma) + np.square(obs_error))
        term2 = np.ones(est_sigma.shape, dtype=np.float32) / (np.square(est_sigma) + np.square(obs_error))
        return np.sum(term1, axis=0) - np.sum(term2, axis=0)

    def evaluate(self, xmap: Xmap, dtag: Dtag):
        xmap_array = np.copy(xmap.to_array())

        residuals = (xmap_array - self.mean)
        denominator = (np.sqrt(np.square(self.sigma_s_m) + np.square(self.sigma_is[dtag])))

        return residuals / denominator

    @staticmethod
    def liklihood(est_sigma, est_mu, obs_vals, obs_error):
        term1 = -np.square(obs_vals - est_mu) / (2 * (np.square(est_sigma) + np.square(obs_error)))
        term2 = np.log(np.ones(est_sigma.shape, dtype=np.float32) / np.sqrt(
            2 * np.pi * (np.square(est_sigma) + np.square(obs_error))))
        return np.sum(term1 + term2)

    @staticmethod
    def log_liklihood_normal(est_sigma, est_mu, obs_vals, obs_error):
        n = obs_error.size

        # term1 = -np.square(obs_vals - est_mu) / (2 * (np.square(est_sigma) + np.square(obs_error)))
        # term2 = np.log(np.ones(est_sigma.shape) / np.sqrt(2 * np.pi * (np.square(est_sigma) + np.square(obs_error))))

        term1 = -1 * (n / 2) * np.log(2 * np.pi)
        term2 = -1 * (1 / 2) * np.log(np.square(est_sigma) + np.square(obs_error))  # n, m
        term3 = -1 * (1 / 2) * (1 / (np.square(est_sigma) + np.square(obs_error))) * np.square(obs_vals - est_mu)  # n,m

        return term1 + np.sum(term2 + term3, axis=0)  # 1 + m

    def save_maps(self, pandda_dir: Path, shell: Shell, p1: bool=True):
        # Mean map
        mean_array = self.mean
        
        grid = gemmi.FloatGrid(*mean_array.shape)
        grid_array = np.array(grid, copy=False)
        mean_array_typed = mean_array.astype(grid_array.dtype)
        
        grid_array[:,:,:] = mean_array_typed[:,:,:]
        
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(0, True)
        ccp4.write_ccp4_map(str(pandda_dir / PANDDA_MEAN_MAP_FILE.format(shell.number, shell.res_min)))
        
        # sigma_s_m map
        sigma_s_m_array = self.sigma_s_m
        
        grid = gemmi.FloatGrid(*sigma_s_m_array.shape)
        grid_array = np.array(grid, copy=False)
        sigma_s_m_array_typed = sigma_s_m_array.astype(grid_array.dtype)
        
        grid_array[:,:,:] = sigma_s_m_array_typed[:,:,:]
        
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(0, True)
        ccp4.write_ccp4_map(str(pandda_dir / PANDDA_SIGMA_S_M_FILE.format(number=shell.number, 
                                                                          res=shell.res_min.resolution,
                                                                          )))
        

@dataclasses.dataclass()
class Zmap:
    zmap: gemmi.FloatGrid

    @staticmethod
    def from_xmap(model: Model, xmap: Xmap, dtag: Dtag):
        zmap_array = model.evaluate(xmap, dtag)
        zmap = Zmap.grid_from_template(xmap, zmap_array)
        return Zmap(zmap)

    @staticmethod
    def grid_from_template(xmap: Xmap, zmap_array: np.array):
        spacing = [xmap.xmap.nu, xmap.xmap.nv, xmap.xmap.nw, ]
        new_grid = gemmi.FloatGrid(*spacing)
        new_grid.spacegroup = xmap.xmap.spacegroup
        new_grid.unit_cell = xmap.xmap.unit_cell

        new_grid_array = np.array(new_grid, copy=False)
        new_grid_array[:, :, :] = zmap_array[:, :, :]

        return new_grid

    def to_array(self, copy=True):
        return np.array(self.zmap, copy=copy)

    def shape(self):
        return [self.zmap.nu, self.zmap.nv, self.zmap.nw]

    def spacegroup(self):
        return self.zmap.spacegroup

    def unit_cell(self):
        return self.zmap.unit_cell

    def save(self, path: Path, p1: bool = True):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = self.zmap
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(path))
        
    def __getstate__(self):
        return XmapPython.from_gemmi(self.zmap)
        
    def __setstate__(self, zmap_pyhton: XmapPython):
        self.zmap = zmap_pyhton.to_gemmi()
        

@dataclasses.dataclass()
class Zmaps:
    zmaps: typing.Dict[Dtag, Zmap]

    @staticmethod
    def from_xmaps(model: Model, xmaps: Xmaps):
        zmaps = {}
        for dtag in xmaps:
            xmap = xmaps[dtag]
            zmap = Zmap.from_xmap(model, xmap, dtag)
            zmaps[dtag] = zmap

        return Zmaps(zmaps)

    def __len__(self):
        return len(self.zmaps)

    def __iter__(self):
        for dtag in self.zmaps:
            yield dtag

    def __getitem__(self, item):
        return self.zmaps[item]


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
class ClusterID:
    dtag: Dtag
    number: EventIDX

    def __hash__(self):
        return hash((self.dtag, self.number))


@dataclasses.dataclass()
class Cluster:
    indexes: typing.Tuple[np.ndarray]
    values: np.ndarray
    centroid: gemmi.Position
    event_mask_indicies: np.ndarray

    def size(self, grid: Grid):
        grid_volume = grid.volume()
        grid_size = grid.size()
        grid_voxel_volume = grid_volume / grid_size
        return self.values.size * grid_voxel_volume

    def peak(self):
        return np.max(self.values)


@dataclasses.dataclass()
class Symops:
    symops: typing.List[gemmi.Op]

    @staticmethod
    def from_grid(grid: gemmi.FloatGrid):
        spacegroup = grid.spacegroup
        operations = list(spacegroup.operations())
        return Symops(operations)

    def __iter__(self):
        for symop in self.symops:
            yield symop

@dataclasses.dataclass()
class Clustering:
    clustering: typing.Dict[int, Cluster]

    @staticmethod
    def from_zmap(zmap: Zmap, reference: Reference, grid: Grid, contour_level: float, cluster_cutoff_distance_multiplier: float=1.3):
        zmap_array = zmap.to_array(copy=True)

        # Get the protein mask
        protein_mask_grid = grid.partitioning.protein_mask
        protein_mask = np.array(protein_mask_grid, copy=False, dtype=np.int8)

        # Get the symmetry mask
        symmetry_contact_mask_grid = grid.partitioning.symmetry_mask
        symmetry_contact_mask = np.array(symmetry_contact_mask_grid, copy=False, dtype=np.int8)

        # Don't consider outlying points away from the protein
        protein_mask_bool = np.full(protein_mask.shape, False)
        protein_mask_bool[np.nonzero(protein_mask)] = True
        zmap_array[~protein_mask_bool] = 0.0

        # Don't consider outlying points at symmetry contacts
        zmap_array[np.nonzero(symmetry_contact_mask)] = 0.0

        extrema_mask_array = zmap_array > contour_level

        # Get the unwrapped coords, and convert them to unwrapped fractional positions, then orthogonal points for clustering       
        point_array = grid.partitioning.coord_array()
        point_tuple = (point_array[:,0],
                       point_array[:,1],
                       point_array[:,2],
                       )
        point_tuple_wrapped = (
            np.mod(point_array[:,0], grid.grid.nu),
            np.mod(point_array[:,1], grid.grid.nv),
            np.mod(point_array[:,2], grid.grid.nw),
                       )
        
        
        extrema_point_mask = extrema_mask_array[point_tuple_wrapped] == 1
        extrema_point_array = point_array[extrema_point_mask]
        extrema_point_wrapped_tuple = (
            point_tuple_wrapped[0][extrema_point_mask],
            point_tuple_wrapped[1][extrema_point_mask],
            point_tuple_wrapped[2][extrema_point_mask],
        )                           
        extrema_fractional_array = extrema_point_array / np.array([grid.grid.nu, grid.grid.nv, grid.grid.nw]).reshape((1,3)) 
        
        positions_orthogonal = [zmap.zmap.unit_cell.orthogonalize(gemmi.Fractional(fractional[0],
                                                                        fractional[1],
                                                                        fractional[2],
                                                                        )) for fractional in extrema_fractional_array]
        positions = [[position.x, position.y, position.z] for position in positions_orthogonal]


        # positions = []
        # for point in extrema_grid_coords_array:
        #     # position = gemmi.Fractional(*point)
        #     point = grid.grid.get_point(*point)
        #     # fractional = grid.grid.point_to_fractional(point)
        #     # pos_orth = zmap.zmap.unit_cell.orthogonalize(fractional)
        #     orthogonal = grid.grid.point_to_position(point)

        #     pos_orth_array = [orthogonal[0],
        #                       orthogonal[1],
        #                       orthogonal[2], ]
        #     positions.append(pos_orth_array)

        extrema_cart_coords_array = np.array(positions)  # n, 3

        point_000 = grid.grid.get_point(0, 0, 0)
        point_111 = grid.grid.get_point(1, 1, 1)
        position_000 = grid.grid.point_to_position(point_000)
        position_111 = grid.grid.point_to_position(point_111)
        clustering_cutoff = position_000.dist(position_111) * cluster_cutoff_distance_multiplier

        if extrema_cart_coords_array.size < 10:
            clusters = {}
            return Clustering(clusters)

        cluster_ids_array = fclusterdata(X=extrema_cart_coords_array,
                                         # t=blob_finding.clustering_cutoff,
                                         t=clustering_cutoff,
                                         criterion='distance',
                                         metric='euclidean',
                                         method='single',
                                         )

        clusters = {}
        for unique_cluster in np.unique(cluster_ids_array):
            if unique_cluster == -1:
                continue
            cluster_mask = cluster_ids_array == unique_cluster  # n
            cluster_indicies = np.nonzero(cluster_mask)  # (n')
            # cluster_points_array = extrema_point_array[cluster_indicies]
            # cluster_points_tuple = (cluster_points_array[:, 0],
            #                         cluster_points_array[:, 1],
            #                         cluster_points_array[:, 2],)
            

            cluster_points_tuple = (
                extrema_point_wrapped_tuple[0][cluster_indicies],
                extrema_point_wrapped_tuple[1][cluster_indicies],
                extrema_point_wrapped_tuple[2][cluster_indicies],
            )

            values = zmap_array[cluster_points_tuple]

            # Generate event mask
            cluster_positions_array = extrema_cart_coords_array[cluster_indicies]
            positions = PositionsArray(cluster_positions_array).to_positions()
            event_mask = gemmi.Int8Grid(*zmap.shape())
            event_mask.spacegroup = zmap.spacegroup()
            event_mask.set_unit_cell(zmap.unit_cell())
            for position in positions:
                event_mask.set_points_around(position,
                                             radius=3.0,
                                             value=1,
                                             )

            # event_mask.symmetrize_max()

            event_mask_array = np.array(event_mask, copy=True, dtype=np.int8)
            event_mask_indicies = np.nonzero(event_mask_array)

            centroid_array = np.mean(cluster_positions_array,
                                     axis=0)
            centroid = (centroid_array[0],
                                      centroid_array[1],
                                      centroid_array[2], )

            cluster = Cluster(cluster_points_tuple,
                              values,
                              centroid,
                              event_mask_indicies,
                              )
            clusters[unique_cluster] = cluster

        return Clustering(clusters)

    def __iter__(self):
        for cluster_num in self.clustering:
            yield cluster_num

    def cluster_mask(self, grid: Grid):
        grid_array = np.array(grid.grid, copy=False)
        mask = gemmi.Int8Grid(*grid_array.shape)
        mask.spacegroup = grid.grid.spacegroup
        mask.set_unit_cell(grid.grid.unit_cell)

        mask_array = np.array(mask, copy=False)

        for cluster_id in self.clustering:
            cluster = self.clustering[cluster_id]
            indexes = cluster.indexes
            mask_array[indexes] = 1

        return mask

    @staticmethod
    def get_protein_mask(zmap: Zmap, reference: Reference, masks_radius: float):
        mask = gemmi.Int8Grid(*zmap.shape())
        mask.spacegroup = zmap.spacegroup()
        mask.set_unit_cell(zmap.unit_cell())

        for atom in reference.dataset.structure.protein_atoms():
            pos = atom.pos
            mask.set_points_around(pos,
                                   radius=masks_radius,
                                   value=1,
                                   )

        mask_array = np.array(mask, copy=False)

        return mask

    # @staticmethod
    # def get_symmetry_contact_mask(zmap: Zmap, reference: Reference, protein_mask: gemmi.Int8Grid,
    #                               symmetry_mask_radius: float = 3):
    #     mask = gemmi.Int8Grid(*zmap.shape())
    #     mask.spacegroup = zmap.spacegroup()
    #     mask.set_unit_cell(zmap.unit_cell())
    #
    #     symops = Symops.from_grid(mask)
    #
    #     for atom in reference.dataset.structure.protein_atoms():
    #         for symmetry_operation in symops.symops[1:]:
    #             position = atom.pos
    #             fractional_position = mask.unit_cell.fractionalize(position)
    #             symmetry_position = gemmi.Fractional(*symmetry_operation.apply_to_xyz([fractional_position[0],
    #                                                                                    fractional_position[1],
    #                                                                                    fractional_position[2],
    #                                                                                    ]))
    #             orthogonal_symmetry_position = mask.unit_cell.orthogonalize(symmetry_position)
    #
    #             mask.set_points_around(orthogonal_symmetry_position,
    #                                    radius=symmetry_mask_radius,
    #                                    value=1,
    #                                    )
    #
    #     mask_array = np.array(mask, copy=False, dtype=np.int8)
    #
    #     protein_mask_array = np.array(protein_mask, copy=False, dtype=np.int8)
    #
    #     equal_mask = protein_mask_array == mask_array
    #
    #     protein_mask_indicies = np.nonzero(protein_mask_array)
    #     protein_mask_bool = np.full(protein_mask_array.shape, False)
    #     protein_mask_bool[protein_mask_indicies] = True
    #
    #     mask_array[~protein_mask_bool] = 0
    #
    #     return mask

    def __getitem__(self, item):
        return self.clustering[item]

    def __len__(self):
        return len(self.clustering)

@dataclasses.dataclass()
class Clusterings:
    clusters: typing.Dict[Dtag, Clustering]

    @staticmethod
    def from_Zmaps(zmaps: Zmaps, reference: Reference, grid: Grid, contour_level: float, cluster_cutoff_distance_multiplier: float,
                   mapper = False):
        
        if mapper:
            keys = list(zmaps.zmaps.keys())
            
            results = mapper(
                                        delayed(
                                            Clustering.from_zmap)(
                                                zmaps[key], 
                                                reference, 
                                                grid, 
                                                contour_level,
                                                cluster_cutoff_distance_multiplier,
                                                )
                                            for key
                                            in keys
                                    )
            clusterings = {keys[i]: results[i]
                for i, key
                in enumerate(keys)
                }
        else:
        
            clusterings = {}
            for dtag in zmaps:
                clustering = Clustering.from_zmap(zmaps[dtag], reference, grid, contour_level)
                clusterings[dtag] = clustering

        return Clusterings(clusterings)

    def filter_size(self, grid: Grid, min_cluster_size: float):
        new_clusterings = {}
        for dtag in self.clusters:
            clustering = self.clusters[dtag]
            new_cluster_nums = list(filter(lambda cluster_num: clustering[cluster_num].size(grid) > min_cluster_size,
                                           clustering,
                                           )
                                    )

            if len(new_cluster_nums) == 0:
                continue

            else:
                new_clusters_dict = {new_cluster_num: clustering[new_cluster_num] for new_cluster_num in
                                     new_cluster_nums}
                new_clustering = Clustering(new_clusters_dict)
                new_clusterings[dtag] = new_clustering

        return Clusterings(new_clusterings)

    def filter_peak(self, grid: Grid, z_peak: float):
        new_clusterings = {}
        for dtag in self.clusters:
            clustering = self.clusters[dtag]
            new_cluster_nums = list(filter(lambda cluster_num: clustering[cluster_num].peak() > z_peak,
                                           clustering,
                                           )
                                    )

            if len(new_cluster_nums) == 0:
                continue

            else:
                new_clusters_dict = {new_cluster_num: clustering[new_cluster_num] for new_cluster_num in
                                     new_cluster_nums}
                new_clustering = Clustering(new_clusters_dict)
                new_clusterings[dtag] = new_clustering

        return Clusterings(new_clusterings)
    
    def merge_clusters(self):
        new_clustering_dict = {}
        for dtag in self.clusters:
            clustering = self.clusters[dtag]
            
            cluster_list = []
            centroid_list = []
            for cluster_id in clustering:
                cluster = clustering[cluster_id]
                centroid = cluster.centroid
                cluster_list.append(cluster)
                centroid_list.append(centroid)
                
            cluster_array = np.array(cluster_list)
            centroid_array = np.array(centroid_list)
            
            dbscan = DBSCAN(
                eps=6,
                min_samples=1,
            )
            
            cluster_ids_array = dbscan.fit_predict(centroid_array)
            new_clusters = {}
            for unique_cluster in np.unique(cluster_ids_array):                
                current_clusters = cluster_array[cluster_ids_array == unique_cluster]
                
                cluster_points_tuple = tuple(np.concatenate([current_cluster.indexes[i] 
                                                  for current_cluster
                                                  in current_clusters
                                                  ], axis=None,
                                                 )
                                        for i
                                        in [0, 1, 2]
                                        )
                cluster_positions_list = [current_cluster.centroid 
                                                  for current_cluster
                                                  in current_clusters
                                                  ]
                
                values = np.concatenate([current_cluster.values 
                                                  for current_cluster
                                                  in current_clusters
                                                  ], axis=None,
                                                 )

                centroid_array = np.mean(np.array(cluster_positions_list), axis=0)
                
                centroid = (centroid_array[0],
                                      centroid_array[1],
                                      centroid_array[2], )                
                event_mask_indicies  = tuple(
                    np.concatenate(
                        [current_cluster.event_mask_indicies[i] 
                         for current_cluster
                         in current_clusters
                         ], 
                        axis=None,
                        )
                    for i
                    in [0, 1, 2]
                    )
                
                
                new_cluster = Cluster(
                    cluster_points_tuple,
                              values,
                              centroid,
                              event_mask_indicies,
                )

                new_clusters[unique_cluster] = new_cluster
                
            new_clustering_dict[dtag] = Clustering(new_clusters)      
            
            
        return Clusterings(new_clustering_dict)      
        

    def filter_distance_from_protein(self):
        pass

    def group_close(self):
        pass

    def remove_symetry_pairs(self):
        pass

    def __getitem__(self, item):
        return self.clusters[item]

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        for dtag in self.clusters:
            yield dtag


@dataclasses.dataclass()
class BDC:
    bdc: float
    fraction: float

    @staticmethod
    def from_float(bdc: float):
        pass

    @staticmethod
    def from_cluster(xmap: Xmap, model: Model, cluster: Cluster, dtag: Dtag, grid: Grid, steps=100):
        xmap_array = xmap.to_array(copy=True)


        cluster_indexes = cluster.event_mask_indicies
        print(f"\t\t\t Cluster indicies of length: {len(cluster_indexes)}")
        print(f"\t\t\t Element length: {(cluster_indexes[0].shape, cluster_indexes[1].shape, cluster_indexes[2].shape,)}")
        


        protein_mask = np.array(grid.partitioning.protein_mask, copy=False, dtype=np.int8)
        protein_mask_indicies = np.nonzero(protein_mask)

        xmap_masked = xmap_array[protein_mask_indicies]
        mean_masked = model.mean[protein_mask_indicies]
        cluster_array = np.full(protein_mask.shape, False)
        cluster_array[cluster_indexes] = True
        cluster_mask = cluster_array[protein_mask_indicies]

        vals = {}
        for val in np.linspace(0, 1, steps):
            subtracted_map = xmap_masked - val * mean_masked
            cluster_vals = subtracted_map[cluster_mask]
            # local_correlation = stats.pearsonr(mean_masked[cluster_mask],
            #                                    cluster_vals)[0]
            local_correlation, local_offset = np.polyfit(x=mean_masked[cluster_mask], y=cluster_vals, deg=1)


            # global_correlation = stats.pearsonr(mean_masked,
            #                                     subtracted_map)[0]
            global_correlation, global_offset = np.polyfit(x=mean_masked, y=subtracted_map, deg=1)


            vals[val] = np.abs(global_correlation - local_correlation)


        fraction = max(vals,
                       key=lambda x: vals[x],
                       )


        return BDC(1 - fraction, fraction)


@dataclasses.dataclass()
class Euclidean3Coord:
    x: float
    y: float
    z: float


@dataclasses.dataclass()
class Site:
    number: int
    centroid: typing.List[float]


@dataclasses.dataclass()
class PositionsArray:
    array: np.ndarray

    @staticmethod
    def from_positions(positions: typing.List[gemmi.Position]):
        accumulator = []
        for position in positions:
            # pos = [position.x, position.y, position.z]
            pos = position
            accumulator.append(pos)

        array = np.array(accumulator)

        return PositionsArray(array)

    def to_array(self):
        return self.array

    def to_positions(self):
        positions = []

        for row in self.array:
            pos = gemmi.Position(row[0],
                                 row[1],
                                 row[2], )
            positions.append(pos)

        return positions


@dataclasses.dataclass()
class SiteID:
    site_id: int

    def __hash__(self):
        return self.site_id


@dataclasses.dataclass()
class Sites:
    site_to_event: typing.Dict[SiteID, typing.List[EventID]]
    event_to_site: typing.Dict[EventID, SiteID]
    centroids: typing.Dict[SiteID, np.ndarray]
    
    def __iter__(self):
        for site_id in self.site_to_event:
            yield site_id
            
    def __getitem__(self, item):
        return self.site_to_event[item]

    @staticmethod
    def from_clusters(clusterings: Clusterings, cutoff: float):
        flat_clusters = {}
        for dtag in clusterings:
            for event_idx in clusterings[dtag]:
                event_idx = EventIDX(event_idx)
                flat_clusters[ClusterID(dtag, event_idx)] = clusterings[dtag][event_idx.event_idx]

        centroids: typing.List[gemmi.Position] = [cluster.centroid for cluster in flat_clusters.values()]
        positions_array = PositionsArray.from_positions(centroids)


        if positions_array.to_array().shape[0] < 4:
            site_to_event = {}
            event_to_site = {}

            for site_id, cluster_id in enumerate(flat_clusters):
                site_id = SiteID(int(site_id))

                if not site_id in site_to_event:
                    site_to_event[site_id] = []

                site_to_event[site_id].append(cluster_id)
                event_to_site[EventID(cluster_id.dtag, cluster_id.number)] = site_id

            return Sites(site_to_event, event_to_site, {})
            

        site_ids_array = fclusterdata(X=positions_array.to_array(),
                                      t=cutoff,
                                      criterion='distance',
                                      metric='euclidean',
                                      method='average',
                                      )

        site_to_event = {}
        event_to_site = {}

        for cluster_id, site_id in zip(flat_clusters, site_ids_array):
            site_id = SiteID(int(site_id))

            if not site_id in site_to_event:
                site_to_event[site_id] = []

            site_to_event[site_id].append(cluster_id)
            event_to_site[EventID(cluster_id.dtag, cluster_id.number)] = site_id
            
        # Site centroids
        site_centroids = {}
        for site_id, event_id_list in site_to_event.items():
            cluster_coord_list = []
            for event_id in event_id_list:
                cluster_centroid = flat_clusters[event_id].centroid
                cluster_coord_list.append(cluster_centroid)
            array = np.array(cluster_coord_list)
            mean_centroid = np.mean(array, axis=0)
            site_centroids[site_id] = mean_centroid

        return Sites(site_to_event, event_to_site, site_centroids)


@dataclasses.dataclass()
class Event:
    event_id: EventID
    site: SiteID
    bdc: BDC
    cluster: Cluster

    @staticmethod
    def from_cluster(event_id: EventID,
                     cluster: Cluster,
                     site: SiteID,
                     bdc: BDC, ):
        return Event(event_id=event_id,
                     site=site,
                     bdc=bdc,
                     cluster=cluster)


@dataclasses.dataclass()
class EventID:
    dtag: Dtag
    event_idx: EventIDX

    def __hash__(self):
        return hash((self.dtag, self.event_idx))


@dataclasses.dataclass()
class Events:
    events: typing.Dict[EventID, Event]
    sites: Sites

    @staticmethod
    def from_clusters(clusterings: Clusterings, model: Model, xmaps: Xmaps, grid: Grid, cutoff: float, mapper: Any=None):
        events: typing.Dict[EventID, Event] = {}

        print(f"\tGetting sites...")
        sites: Sites = Sites.from_clusters(clusterings, cutoff)

        if mapper:
            jobs = {}
            for dtag in clusterings:
                clustering = clusterings[dtag]
                for event_idx in clustering:
                    event_idx = EventIDX(event_idx)
                    event_id = EventID(dtag, event_idx)
                    
                    cluster = clustering[event_idx.event_idx]
                    xmap = xmaps[dtag]
                    
                    site: SiteID = sites.event_to_site[event_id]


                    jobs[event_id] = delayed(Events.get_event)(xmap, cluster, dtag, site, event_id, model, grid)

            
            results = mapper(job for job in jobs.values())
                   
            events = {event_id: event for event_id, event in zip(jobs.keys(), results)}
            
        else:
            for dtag in clusterings:
                clustering = clusterings[dtag]
                for event_idx in clustering:
                    print(f"\t\tGetting bdc...")
                    event_idx = EventIDX(event_idx)
                    event_id = EventID(dtag, event_idx)

                    cluster = clustering[event_idx.event_idx]
                    xmap = xmaps[dtag]
                    bdc = BDC.from_cluster(xmap, model, cluster, dtag, grid)

                    site: SiteID = sites.event_to_site[event_id]

                    event = Event.from_cluster(event_id,
                                            cluster,
                                            site,
                                            bdc,
                                            )

                    events[event_id] = event

        return Events(events, sites)
    
    @staticmethod
    def get_event(xmap, cluster, dtag, site, event_id, model, grid):
        bdc = BDC.from_cluster(xmap, model, cluster, dtag, grid,)

        event = Event.from_cluster(
            event_id,
            cluster,
            site,
            bdc,
            )
        
        return event

    @staticmethod
    def from_all_events(event_dict: typing.Dict[EventID, Event], grid: Grid, cutoff: float):

        all_clusterings_dict = {}
        for event_id in event_dict:
            if event_id.dtag not in all_clusterings_dict:
                all_clusterings_dict[event_id.dtag] = {}
                
            all_clusterings_dict[event_id.dtag][event_id.event_idx.event_idx] = event_dict[event_id].cluster

        all_clusterings = {}
        for dtag in all_clusterings_dict:
            all_clusterings[dtag] = Clustering(all_clusterings_dict[dtag])
            
        clusterings = Clusterings(all_clusterings)

        sites: Sites = Sites.from_clusters(clusterings, cutoff)

        events: typing.Dict[EventID, Event] = {}
        for event_id in event_dict:
            event = event_dict[event_id]
            
            for event_id_site, event_site in sites.event_to_site.items():
                if (event_id_site.dtag.dtag == event_id.dtag.dtag) and (event_id_site.event_idx.event_idx == event_id.event_idx.event_idx):
                    site = event_site
                    
            event.site = site

            events[event_id] = event

        return Events(events, sites)

    def __iter__(self):
        for event_id in self.events:
            yield event_id

    def __getitem__(self, item):
        return self.events[item]
    
    def save_event_maps(
        self, 
        datasets, 
        alignments, 
        xmaps,
        model, 
        pandda_fs_model,
        grid,
        structure_factors, 
        outer_mask,
        inner_mask_symmetry,
        mapper=False,
        ):
        
        processed_datasets = {}
        for event_id in self:
            dtag = event_id.dtag
            event = self[event_id]
            string = f"""
            dtag: {dtag}
            event bdc: {event.bdc}
            centroid: {event.cluster.centroid}
            """
            if dtag not in processed_datasets:
                processed_datasets[dtag] = pandda_fs_model.processed_datasets[event_id.dtag]
            
            processed_datasets[dtag].event_map_files.add_event(event)
            
        if mapper:
            event_id_list = list(self.events.keys())
            
            results = mapper(
                    delayed(
                        processed_datasets[event_id.dtag].event_map_files[event_id.event_idx].save)(
                            xmaps[event_id.dtag],
                            model,
                            self[event_id],
                            datasets[event_id.dtag], 
                            alignments[event_id.dtag],
                            grid, 
                            structure_factors, 
                            outer_mask,
                            inner_mask_symmetry,
                            )
                        for event_id
                        in event_id_list
                        )
                                    




@dataclasses.dataclass()
class ZMapFile:
    path: Path

    @staticmethod
    def from_zmap(zmap: Zmap):
        pass

    @staticmethod
    def from_dir(path: Path, dtag: str):
        return ZMapFile(path / PANDDA_Z_MAP_FILE.format(dtag=dtag))

    def save(self, zmap: Zmap):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = zmap.zmap
        ccp4.update_ccp4_header(2, True)
        ccp4.grid.symmetrize_max()
        ccp4.write_ccp4_map(str(self.path))


# @dataclasses.dataclass()
# class ZMapFiles:
#
#     @staticmethod
#     def from_zmaps(zmaps: Zmaps, pandda_fs_model: PanDDAFSModel):
#         for dtag in zmaps:
#             processed_dataset = pandda_fs_model.processed_datasets[dtag]
#
#             zmaps[dtag].save()




@dataclasses.dataclass()
class EventMapFile:
    path: Path

    @staticmethod
    def from_event(event: Event, path: Path):
        rounded_bdc = round(event.bdc.bdc, 2)
        event_map_path = path / PANDDA_EVENT_MAP_FILE.format(dtag=event.event_id.dtag.dtag,
                                                             event_idx=event.event_id.event_idx.event_idx,
                                                             bdc=rounded_bdc,
                                                             )
        return EventMapFile(event_map_path)

    def save(self, 
             xmap: Xmap,
             model: Model,
             event: Event,
             dataset: Dataset,
             alignment: Alignment,
             grid: Grid,
             structure_factors: StructureFactors,
             mask_radius: float,
             mask_radius_symmetry: float,
             ):
        reference_xmap_grid = xmap.xmap
        reference_xmap_grid_array = np.array(reference_xmap_grid, copy=True)

        # moving_xmap_grid: gemmi.FloatGrid = dataset.reflections.reflections.transform_f_phi_to_map(structure_factors.f,
        #                                                                                          structure_factors.phi,
        #                                                                                          )

        event_map_reference_grid = gemmi.FloatGrid(*[reference_xmap_grid.nu,
                                                     reference_xmap_grid.nv,
                                                     reference_xmap_grid.nw,
                                                     ]
                                                   )
        event_map_reference_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")  # xmap.xmap.spacegroup
        event_map_reference_grid.set_unit_cell(reference_xmap_grid.unit_cell)

        event_map_reference_grid_array = np.array(event_map_reference_grid,
                                                  copy=False,
                                                  )

        mean_array = model.mean
        event_map_reference_grid_array[:, :, :] = (reference_xmap_grid_array - (event.bdc.bdc * mean_array)) / (
                1 - event.bdc.bdc)

        event_map_grid = Xmap.from_aligned_map_c(event_map_reference_grid,
                                               dataset,
                                               alignment,
                                               grid,
                                               structure_factors,
                                               mask_radius,
                                               mask_radius_symmetry,
                                               )

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = event_map_grid.xmap
        ccp4.update_ccp4_header(2, True)
        # ccp4.grid.symmetrize_max()
        ccp4.write_ccp4_map(str(self.path))


@dataclasses.dataclass()
class EventMapFiles:
    path: Path
    event_map_files: typing.Dict[EventIDX, EventMapFile]

    # @staticmethod
    # def from_events(events: Events, xmaps: Xmaps):
    #     pass

    @staticmethod
    def from_dir(dir: Path):
        return EventMapFiles(dir, {})

    def get_events(self, events: typing.Dict[EventIDX, Event]):
        event_map_files = {}
        for event_idx in events:
            event_map_files[event_idx] = EventMapFile.from_event(events[event_idx], self.path)

        self.event_map_files = event_map_files

    def add_event(self, event: Event):
        self.event_map_files[event.event_id.event_idx] = EventMapFile.from_event(event, self.path)

    def __iter__(self):
        for event_idx in self.event_map_files:
            yield event_idx

    def __getitem__(self, item):
        return self.event_map_files[item]


@dataclasses.dataclass()
class EventTableRecord:
    dtag:str
    event_idx: int	
    bdc: float	
    cluster_size: int	
    global_correlation_to_average_map: float	
    global_correlation_to_mean_map: float	
    local_correlation_to_average_map: float	
    local_correlation_to_mean_map: float	
    site_idx: int	
    x: float	
    y: float	
    z: float	
    z_mean: float	
    z_peak: float	
    applied_b_factor_scaling: float	
    high_resolution: float	
    low_resolution: float	
    r_free: float	
    r_work: float
    analysed_resolution: float
    map_uncertainty: float
    analysed: bool	
    interesting: bool	
    exclude_from_z_map_analysis: bool	
    exclude_from_characterisation: bool

    @staticmethod
    def from_event(event: Event):
        return EventTableRecord(
            dtag=event.event_id.dtag.dtag,
            event_idx=event.event_id.event_idx.event_idx,
            bdc=event.bdc.bdc,
            cluster_size=event.cluster.values.size,
            global_correlation_to_average_map=0,
            global_correlation_to_mean_map=0,
            local_correlation_to_average_map=0,
            local_correlation_to_mean_map=0,
            site_idx=event.site.site_id,
            x=event.cluster.centroid[0],
            y=event.cluster.centroid[1],
            z=event.cluster.centroid[2],
            z_mean=0.0,
            z_peak=0.0,
            applied_b_factor_scaling=0.0,
            high_resolution=0.0,
            low_resolution=0.0,
            r_free=0.0,
            r_work=0.0,
            analysed_resolution=0.0,
            map_uncertainty=0.0,
            analysed=False,
            interesting=False,	
            exclude_from_z_map_analysis=False,	
            exclude_from_characterisation=False,
                                )
        
@dataclasses.dataclass()
class EventTable:
    records: List[EventTableRecord]
    
    @staticmethod
    def from_events(events: Events):
        records = []
        for event_id in events:
            event_record = EventTableRecord.from_event(events[event_id])
            records.append(event_record)
        
        return EventTable(records)
    
    def save(self, path: Path):
        records = []
        for record in self.records:
            event_dict = dataclasses.asdict(record) 
            event_dict["1-BDC"] = round(event_dict["bdc"], 2)
            records.append(event_dict)
        table = pd.DataFrame(records)
        table.to_csv(str(path))
        
        
@dataclasses.dataclass()
class SiteTableRecord:
    site_idx: int	
    centroid: Tuple[float, float, float]
    
    @staticmethod
    def from_site_id(site_id: SiteID, centroid: np.ndarray):
        return SiteTableRecord(
            site_idx=site_id.site_id,
            centroid=(centroid[0], centroid[1], centroid[2],), 
        )
        

@dataclasses.dataclass()
class SiteTable:
    site_record_list: List[SiteTableRecord]

    def __iter__(self):
        for record in self.site_record_list:
            yield record

    @staticmethod
    def from_events(events: Events, cutoff: float):
        
        dtag_clusters = {}
        for event_id in events:
            dtag = event_id.dtag
            event_idx = event_id.event_idx
            event = events[event_id]
            
            if dtag not in dtag_clusters:
                dtag_clusters[dtag] = {}
            
            dtag_clusters[dtag][event_idx] = event.cluster
            
        _clusterings = {}
        for dtag in dtag_clusters:
            _clusterings[dtag] = Clustering(dtag_clusters[dtag])
            
        clusterings = Clusterings(_clusterings)
            
        sites: Sites = Sites.from_clusters(clusterings, cutoff)
        
        records = []
        for site_id in sites:
            # site = sites[site_id]
            centroid = sites.centroids[site_id]
            site_record = SiteTableRecord.from_site_id(site_id, centroid)
            records.append(site_record)
        
        return SiteTable(records)
        
        
        
    def save(self, path: Path):
        records = []
        for site_record in self.site_record_list:
            site_record_dict = dataclasses.asdict(site_record)
            records.append(site_record_dict)
            
        table = pd.DataFrame(records)
        
        table.to_csv(str(path))
        



@dataclasses.dataclass()
class SiteTableFile:

    @staticmethod
    def from_events(events: Events):
        pass


@dataclasses.dataclass()
class EventTableFile:

    @staticmethod
    def from_events(events: Events):
        pass


@dataclasses.dataclass()
class Analyses:
    analyses_dir: Path
    pandda_analyse_events_file: Path
    pandda_analyse_sites_file: Path


    @staticmethod
    def from_pandda_dir(pandda_dir: Path):
        analyses_dir = pandda_dir / PANDDA_ANALYSES_DIR
        pandda_analyse_events_file = analyses_dir / PANDDA_ANALYSE_EVENTS_FILE
        pandda_analyse_sites_file = analyses_dir / PANDDA_ANALYSE_SITES_FILE

        return Analyses(analyses_dir=analyses_dir,
                        pandda_analyse_events_file=pandda_analyse_events_file,
                        pandda_analyse_sites_file=pandda_analyse_sites_file,
                        )

    def build(self):
        if not self.analyses_dir.exists():
            os.mkdir(str(self.analyses_dir))


@dataclasses.dataclass()
class DatasetModels:
    path: Path

    @staticmethod
    def from_dir(path: Path):
        return DatasetModels(path=path)


@dataclasses.dataclass()
class LigandDir:
    path: Path
    pdbs: typing.List[Path]
    cifs: typing.List[Path]
    smiles: typing.List[Path]

    @staticmethod
    def from_path(path: Path):
        pdbs = list(path.glob("*.pdb"))
        cifs = list(path.glob("*.cifs"))
        smiles = list(path.glob("*.smiles"))

        return LigandDir(path,
                         pdbs,
                         cifs,
                         smiles,
                         )


@dataclasses.dataclass()
class DatasetDir:
    input_pdb_file: Path
    input_mtz_file: Path
    ligand_dir: Union[LigandDir, None]
    source_ligand_cif: Union[Path, None]
    source_ligand_pdb: Union[Path, None]

    @staticmethod
    def from_path(path: Path, pdb_regex: str, mtz_regex: str, ligand_cif_regex: str, ligand_pdb_regex: str):
        input_pdb_file: Path = next(path.glob(pdb_regex))
        input_mtz_file: Path = next(path.glob(mtz_regex))
        
        source_ligand_dir = path / PANDDA_LIGAND_FILES_DIR
        if source_ligand_dir.exists():
            ligand_dir = LigandDir.from_path(source_ligand_dir)
        else:
            ligand_dir = None
        
        try:
            ligands = path.rglob(ligand_cif_regex)
            source_ligand_cif = next(ligands)
        except:
            source_ligand_cif = None

        try:
            ligands = path.rglob(ligand_pdb_regex)
            source_ligand_pdb = next(ligands)
        except:
            source_ligand_pdb = None

        return DatasetDir(input_pdb_file=input_pdb_file,
                          input_mtz_file=input_mtz_file,
                          ligand_dir=ligand_dir,
                          source_ligand_cif=source_ligand_cif,
                          source_ligand_pdb=source_ligand_pdb,
                          )


@dataclasses.dataclass()
class DataDirs:
    dataset_dirs: typing.Dict[Dtag, DatasetDir]

    @staticmethod
    def from_dir(directory: Path, pdb_regex: str, mtz_regex: str, ligand_cif_regex: str, ligand_pdb_regex: str):
        dataset_dir_paths = list(directory.glob("*"))

        dataset_dirs = {}

        for dataset_dir_path in dataset_dir_paths:
            dtag = Dtag(dataset_dir_path.name)
            try:
                dataset_dir = DatasetDir.from_path(dataset_dir_path, pdb_regex, mtz_regex, ligand_cif_regex, ligand_pdb_regex)
                dataset_dirs[dtag] = dataset_dir
            except:
                continue 

        return DataDirs(dataset_dirs)

    def to_dict(self):
        return self.dataset_dirs


@dataclasses.dataclass()
class ProcessedDataset:
    path: Path
    dataset_models: DatasetModels
    input_mtz: Path
    input_pdb: Path
    source_mtz: Path
    source_pdb: Path
    z_map_file: ZMapFile
    event_map_files: EventMapFiles
    source_ligand_cif: Union[Path, None]
    source_ligand_pdb: Union[Path, None]
    input_ligand_cif: Path
    input_ligand_pdb: Path
    source_ligand_dir: Union[LigandDir, None]
    input_ligand_dir: Path

    @staticmethod
    def from_dataset_dir(dataset_dir: DatasetDir, processed_dataset_dir: Path) -> ProcessedDataset:
        dataset_models_dir = processed_dataset_dir / PANDDA_MODELLED_STRUCTURES_DIR
        
        
        # Copy the input pdb and mtz
        dtag = processed_dataset_dir.name
        source_mtz = dataset_dir.input_mtz_file
        source_pdb = dataset_dir.input_pdb_file
        source_ligand_cif = dataset_dir.source_ligand_cif
        source_ligand_pdb = dataset_dir.source_ligand_pdb
        
        
        input_mtz = processed_dataset_dir / PANDDA_MTZ_FILE.format(dtag)
        input_pdb = processed_dataset_dir / PANDDA_PDB_FILE.format(dtag)
        input_ligand_cif = processed_dataset_dir / PANDDA_LIGAND_CIF_FILE
        input_ligand_pdb = processed_dataset_dir / PANDDA_LIGAND_PDB_FILE
        
        
        z_map_file = ZMapFile.from_dir(processed_dataset_dir, processed_dataset_dir.name)
        event_map_files = EventMapFiles.from_dir(processed_dataset_dir)
        
        source_ligand_dir = dataset_dir.ligand_dir
        input_ligand_dir = processed_dataset_dir / PANDDA_LIGAND_FILES_DIR
        

        return ProcessedDataset(
            path=processed_dataset_dir,
            dataset_models=DatasetModels.from_dir(dataset_models_dir),
            input_mtz=input_mtz,
            input_pdb=input_pdb,
            source_mtz=source_mtz,
            source_pdb=source_pdb,
            z_map_file=z_map_file,
            event_map_files=event_map_files,
            source_ligand_cif=source_ligand_cif,
            source_ligand_pdb=source_ligand_pdb,
            input_ligand_cif=input_ligand_cif,
            input_ligand_pdb=input_ligand_pdb,
            source_ligand_dir=source_ligand_dir,
            input_ligand_dir=input_ligand_dir,
            )

    def build(self):
        if not self.path.exists():
            os.mkdir(str(self.path))

        shutil.copyfile(self.source_mtz, self.input_mtz)
        shutil.copyfile(self.source_pdb, self.input_pdb)
        
        if self.source_ligand_cif: shutil.copyfile(self.source_ligand_cif, self.input_ligand_cif)
        if self.source_ligand_pdb: shutil.copyfile(self.source_ligand_pdb, self.input_ligand_pdb)
        
        if self.source_ligand_dir: 
            shutil.copytree(str(self.source_ligand_dir.path),
                                            str(self.input_ligand_dir),
                        )
        
        
        
@dataclasses.dataclass()
class ProcessedDatasets:
    path: Path
    processed_datasets: typing.Dict[Dtag, ProcessedDataset]

    @staticmethod
    def from_data_dirs(data_dirs: DataDirs, processed_datasets_dir: Path):
        processed_datasets = {}
        for dtag, dataset_dir in data_dirs.dataset_dirs.items():
            processed_datasets[dtag] = ProcessedDataset.from_dataset_dir(dataset_dir,
                                                                         processed_datasets_dir / dtag.dtag,
                                                                         )

        return ProcessedDatasets(processed_datasets_dir,
                                 processed_datasets)

    def __getitem__(self, item):
        return self.processed_datasets[item]

    def __iter__(self):
        for dtag in self.processed_datasets:
            yield dtag

    def build(self):
        if not self.path.exists():
            os.mkdir(str(self.path))

        for dtag in self.processed_datasets:
            self.processed_datasets[dtag].build()


@dataclasses.dataclass()
class PanDDAFSModel:
    pandda_dir: Path
    data_dirs: DataDirs
    analyses: Analyses
    processed_datasets: ProcessedDatasets
    log_file: Path

    @staticmethod
    def from_dir(input_data_dirs: Path,
                 output_out_dir: Path,
                 pdb_regex: str, mtz_regex: str,
                 ligand_cif_regex: str, ligand_pdb_regex: str,
                 ):
        analyses = Analyses.from_pandda_dir(output_out_dir)
        data_dirs = DataDirs.from_dir(input_data_dirs, pdb_regex, mtz_regex, ligand_cif_regex, ligand_pdb_regex)
        processed_datasets = ProcessedDatasets.from_data_dirs(data_dirs,
                                                              output_out_dir / PANDDA_PROCESSED_DATASETS_DIR,
                                                              )
        log_path = output_out_dir / PANDDA_LOG_FILE

        return PanDDAFSModel(pandda_dir=output_out_dir,
                             data_dirs=data_dirs,
                             analyses=analyses,
                             processed_datasets=processed_datasets,
                             log_file=log_path,
                             )

    def build(self, overwrite=False):
        if not self.pandda_dir.exists():
            os.mkdir(str(self.pandda_dir))

        self.processed_datasets.build()
        self.analyses.build()


@dataclasses.dataclass()
class RMSD:
    rmsd: float

    @staticmethod
    def from_structures(structure_1: Structure, structure_2: Structure):

                

        distances = []

        positions_1 = []
        positions_2 = []


        for residues_id in structure_1.protein_residue_ids():
            
            res_1 = structure_1[residues_id][0]
            try:
                res_2 = structure_2[residues_id][0]
            except:
                continue


            res_1_ca = res_1["CA"][0]
            res_2_ca = res_2["CA"][0]

            res_1_ca_pos = res_1_ca.pos
            res_2_ca_pos = res_2_ca.pos


            positions_1.append(res_1_ca_pos)
            positions_2.append(res_2_ca_pos)

            distances.append(res_1_ca_pos.dist(res_2_ca_pos))

        positions_1_array = np.array([[x[0], x[1], x[2]] for x in positions_1])
        positions_2_array = np.array([[x[0], x[1], x[2]] for x in positions_2])


        return RMSD.from_arrays(positions_1_array, positions_2_array)


    @staticmethod
    def from_arrays(array_1, array_2):

        
        array_1_mean = np.mean(array_1, axis=0).reshape((1, 3))
        array_2_mean = np.mean(array_2, axis=0).reshape((1, 3))



        array_1_demeaned = array_1 - array_1_mean
        array_2_demeaned = array_2 - array_2_mean
        #

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(array_1_demeaned, array_2_demeaned)
        rotated_vecs = rotation.apply(array_2_demeaned)

        #
        true_rmsd = np.sqrt(
            np.sum(np.square(np.linalg.norm(array_1_demeaned - rotated_vecs, axis=1)), axis=0) / array_1.shape[0])


        return RMSD(true_rmsd)

    def to_float(self):
        return self.rmsd


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
class DaskMapper:
    cluster: Any
    mapper: Any
    address: str
    
    @staticmethod
    def initialise():
        cluster = LocalCluster()
        cluster.scale(10)
        client = Client()
        address = client.scheduler_info()['services']
        return DaskMapper(cluster, client, address) 
    
    def __call__(self, iterable) -> Any:
        futures = []
        for func in iterable:
            future = self.mapper.submit(func)
            futures.append(future)
            
        results = [future.result() for future in futures]
        
        return results 

@dataclasses.dataclass()
class JoblibMapper:
    mapper: Any
    
    @staticmethod
    def initialise():
        mapper = joblib.Parallel(n_jobs=-2, 
                                      verbose=15,
                                      backend="loky",
                                       max_nbytes=None,
                                       )
        return JoblibMapper(mapper) 
    
    def __call__(self, iterable) -> Any:
        results = self.mapper(joblib.delayed(x)() for x in iterable)
        
        return results
    