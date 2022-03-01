from __future__ import annotations

import typing
from typing import Tuple
import dataclasses
from pathlib import Path
from functools import partial

import scipy
from scipy import spatial
from joblib.externals.loky import set_loky_pickler

set_loky_pickler('pickle')

from sklearn import neighbors

import pandas as pd
import ray

from pandda_gemmi.constants import *
from pandda_gemmi.python_types import *
from pandda_gemmi.common import Dtag, delayed
from pandda_gemmi.common import Partial


# from pandda_gemmi.fs import PanDDAFSModel
# from pandda_gemmi.dataset import (StructureFactors, Structure, Reflections, Dataset, ResidueID, Datasets,
#                                   Resolution, Reference)
# from pandda_gemmi.edalignment import Alignment, Alignments, Transform, Grid, Xmap

@dataclasses.dataclass()
class Resolution:
    resolution: float

    @staticmethod
    def from_float(res: float):
        return Resolution(res)

    def to_float(self) -> float:
        return self.resolution


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
        try:
            structure = gemmi.read_structure(str(file))
        except Exception as e:
            raise Exception(f'Error trying to open file: {file}: {e}')
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

                    try:
                        has_ca = residue["CA"][0]
                    except Exception as e:
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

    def all_atoms(self, exclude_waters=False):
        if exclude_waters:

            for model in self.structure:
                for chain in model:
                    for residue in chain:
                        if residue.is_water():
                            continue

                        for atom in residue:
                            yield atom

        else:
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
class Reflections:
    reflections: gemmi.Mtz
    path: typing.Union[Path, None] = None

    @staticmethod
    def from_file(file: Path) -> Reflections:

        try:
            reflections = gemmi.read_mtz_file(str(file))
        except Exception as e:
            raise Exception(f'Error trying to open file: {file}: {e}')
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

    def drop_columns(self, structure_factors: StructureFactors):
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = self.reflections.spacegroup
        new_reflections.set_cell_for_all(self.reflections.cell)

        # Add dataset
        new_reflections.add_dataset("truncated")

        free_flag = None

        for column in self.reflections.columns:
            if column.label == "FREE":
                free_flag = "FREE"
                break
            if column.label == "FreeR_flag":
                free_flag = "FreeR_flag"
                break

        if not free_flag:
            raise Exception("No RFree Flag found!")

        # Add columns
        for column in self.reflections.columns:
            if column.label in ["H", "K", "L", free_flag, structure_factors.f, structure_factors.phi]:
                new_reflections.add_column(column.label, column.type)

        # Get data
        data_array = np.array(self.reflections, copy=True)
        data = pd.DataFrame(data_array,
                            columns=self.reflections.column_labels(),
                            )
        data.set_index(["H", "K", "L"], inplace=True)

        # Truncate by columns
        data_indexed = data[[free_flag, structure_factors.f, structure_factors.phi]]

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
class Reference:
    dtag: Dtag
    dataset: Dataset

    # @staticmethod
    # def assert_from_datasets(datasets: Datasets):
    #     if len(datasets) < 1:
    #          raise pandda_exceptions.ExceptionTooFewDatasets()

    @staticmethod
    def from_datasets(datasets: Datasets):
        # Reference.assert_from_datasets(datasets)

        resolutions: typing.Dict[Dtag, Resolution] = {}
        for dtag in datasets:
            resolutions[dtag] = datasets[dtag].reflections.resolution()

        min_resolution_dtag = min(
            resolutions,
            key=lambda dtag: resolutions[dtag].to_float(),
        )

        min_resolution_structure = datasets[min_resolution_dtag].structure
        min_resolution_reflections = datasets[min_resolution_dtag].reflections

        return Reference(min_resolution_dtag,
                         datasets[min_resolution_dtag]
                         )


@dataclasses.dataclass()
class Dataset:
    structure: Structure
    reflections: Reflections
    smoothing_factor: float = 0.0

    @staticmethod
    def from_files(pdb_file: Path, mtz_file: Path, ):
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
                       self.reflections.truncate_reflections(index,
                                                             )
                       )

    def scale_reflections(self, reference: Reference):
        new_reflections = self.reflections.scale_reflections(reference)
        return Dataset(self.structure,
                       new_reflections,
                       )

    def drop_columns(self, structure_factors: StructureFactors):
        new_reflections = self.reflections.drop_columns(structure_factors)

        return Dataset(self.structure,
                       new_reflections)

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
        dtag_flattened_index = dtag_reflections_table[
            ~dtag_reflections_table[structure_factors.f].isna()].index.to_flat_index()

        # Get reference
        reference_reflections = reference_ref.reflections
        reference_reflections_array = np.array(reference_reflections, copy=True)
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                                   columns=reference_reflections.column_labels(),
                                                   )
        reference_reflections_table.set_index(["H", "K", "L"], inplace=True)
        reference_flattened_index = reference_reflections_table[
            ~reference_reflections_table[structure_factors.f].isna()].index.to_flat_index()

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
        knn_x.fit(r.reshape(-1, 1),
                  x.reshape(-1, 1),
                  )
        x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)

        scales = []
        rmsds = []

        knn_y = neighbors.RadiusNeighborsRegressor(0.01)
        knn_y.fit(r.reshape(-1, 1),
                  (y * np.exp(0.0 * r)).reshape(-1, 1),
                  )

        # y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

        y_neighbours = knn_y.radius_neighbors(sample_grid[:, np.newaxis])

        # Optimise the scale factor
        for scale in np.linspace(-4, 4, 100):
            y_s = y * np.exp(scale * r)
            # knn_y = neighbors.RadiusNeighborsRegressor(0.01)
            # knn_y.fit(r.reshape(-1, 1),
            #           y_s.reshape(-1, 1),
            #           )
            #
            # y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

            y_f = np.array([np.mean(y_s[y_neighbours[1][j]]) for j, val in enumerate(sample_grid[:,
                                                                                     np.newaxis].flatten())])

            rmsd = np.sum(np.abs(x_f - y_f))

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

        f_scaled_array = f_array * np.exp(min_scale * original_reflections.make_1_d2_array())

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

    # def correlation(self,
    #                 alignment: Alignment,
    #                 grid: Grid,
    #                 structure_factors,
    #                 reference_array: np.ndarray,
    #                 sample_rate: float = 3.0,
    #                 ) -> float:
    #
    #     # Get aligned dataset
    #     xmap: Xmap = Xmap.from_unaligned_dataset_c(
    #         self,
    #         alignment,
    #         grid,
    #         structure_factors,
    #         sample_rate,
    #     )
    #
    #     # Get arrays
    #     xmap_array: np.ndarray = xmap.to_array()
    #
    #     # Get correlation
    #     correlation, map_off = np.polyfit(x=xmap_array.flatten(),
    #                                       y=reference_array.flatten(),
    #                                       deg=1,
    #                                       )
    #
    #     # return
    #     return correlation

def smooth(dataset, reference: Reference, structure_factors: StructureFactors):
    reference_dataset = reference.dataset

    # Get common set of reflections
    common_reflections = dataset.common_reflections(reference_dataset.reflections,
                                                 structure_factors,
                                                 )

    # Truncate
    truncated_reference = reference.dataset.truncate_reflections(common_reflections)
    truncated_dataset = dataset.truncate_reflections(common_reflections)

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
    knn_x.fit(r.reshape(-1, 1),
              x.reshape(-1, 1),
              )
    x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)

    scales = []
    rmsds = []

    knn_y = neighbors.RadiusNeighborsRegressor(0.01)
    knn_y.fit(r.reshape(-1, 1),
              (y * np.exp(0.0 * r)).reshape(-1, 1),
              )

    # y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

    y_neighbours = knn_y.radius_neighbors(sample_grid[:, np.newaxis])

    # Optimise the scale factor
    for scale in np.linspace(-4, 4, 100):
        y_s = y * np.exp(scale * r)
        # knn_y = neighbors.RadiusNeighborsRegressor(0.01)
        # knn_y.fit(r.reshape(-1, 1),
        #           y_s.reshape(-1, 1),
        #           )
        #
        # y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

        y_f = np.array([np.mean(y_s[y_neighbours[1][j]]) for j, val in enumerate(sample_grid[:,
                                                                                 np.newaxis].flatten())])

        rmsd = np.sum(np.abs(x_f - y_f))

        scales.append(scale)
        rmsds.append(rmsd)

    min_scale = scales[np.argmin(rmsds)]

    # Get the original reflections
    original_reflections = dataset.reflections.reflections

    original_reflections_array = np.array(original_reflections,
                                          copy=True,
                                          )

    original_reflections_table = pd.DataFrame(original_reflections_array,
                                              columns=reference_reflections.column_labels(),
                                              )

    f_array = original_reflections_table[structure_factors.f]

    f_scaled_array = f_array * np.exp(min_scale * original_reflections.make_1_d2_array())

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
    smoothed_dataset = Dataset(dataset.structure,
                               Reflections(new_reflections),
                               )

    return smoothed_dataset

@ray.remote
def smooth_ray(dataset, reference: Reference, structure_factors: StructureFactors):
    return smooth(dataset, reference, structure_factors)

@dataclasses.dataclass()
class RMSD:
    rmsd: float

    @staticmethod
    def from_reference(reference: Reference, dataset: Dataset):
        return RMSD.from_structures(
            reference.dataset.structure,
            dataset.structure,

        )

    @staticmethod
    def from_structures(structure_1: Structure, structure_2: Structure, ) -> RMSD:

        distances = []

        positions_1 = []
        positions_2 = []

        # for residues_id in structure_1.protein_residue_ids():
        for residues_id in structure_1.protein_residue_ids():

            res_1 = structure_1[residues_id][0]
            try:
                res_2 = structure_2[residues_id][0]
            except:
                continue

            # print(f"Residue 1 is: {res_1}")
            # print(f"Residue 2 is: {res_2}")
            try:
                res_1_ca = res_1["CA"][0]
            except Exception as e:
                continue

            try:
                res_2_ca = res_2["CA"][0]
            except Exception as e:
                continue

            res_1_ca_pos = res_1_ca.pos
            res_2_ca_pos = res_2_ca.pos

            positions_1.append(res_1_ca_pos)
            positions_2.append(res_2_ca_pos)

            distances.append(res_1_ca_pos.dist(res_2_ca_pos))

        positions_1_array = np.array([[x[0], x[1], x[2]] for x in positions_1])
        positions_2_array = np.array([[x[0], x[1], x[2]] for x in positions_2])

        if positions_1_array.size < 3:
            return RMSD(100.0)
        if positions_2_array.size < 3:
            return RMSD(100.0)

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
class Datasets:
    datasets: typing.Dict[Dtag, Dataset]

    def __len__(self):
        return len(self.datasets)

    @staticmethod
    def from_data_dirs(path: Path, pdb_regex: str, mtz_regex: str):
        datasets = {}
        for directory in path.glob("*"):
            if directory.is_dir():
                dtag = Dtag(directory.name)

                mtz_files = list(directory.glob(mtz_regex))
                if len(mtz_files) == 0:
                    continue

                pdb_files = list(directory.glob(pdb_regex))
                if len(pdb_files) == 0:
                    continue

                datasets[dtag] = Dataset.from_files(pdb_files[0],
                                                    mtz_files[0])
        return Datasets(datasets)

    @staticmethod
    def from_dir(pandda_fs_model  #: PanDDAFSModel,
                 ):
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

        new_dtags = filter(lambda dtag: (RMSD.from_reference(
            reference,
            self.datasets[dtag],
        )).to_float() < max_rmsd_to_reference,
                           self.datasets,
                           )

        new_datasets = {dtag: self.datasets[dtag] for dtag in new_dtags}

        return Datasets(new_datasets)

    def remove_invalid_structure_factor_datasets(self,
                                                 structure_factors: StructureFactors,
                                                 ) -> Datasets:

        new_dtags = filter(
            lambda dtag: (structure_factors.f in self.datasets[dtag].reflections.columns()) and (
                    structure_factors.phi in self.datasets[dtag].reflections.columns()),
            self.datasets,
        )

        new_datasets = {dtag: self.datasets[dtag] for dtag in new_dtags}

        return Datasets(new_datasets)

    def drop_columns(self, structure_factors: StructureFactors):
        new_datasets = {dtag: self.datasets[dtag].drop_columns(structure_factors) for dtag in self.datasets}

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

    def common_reflections(self, structure_factors: StructureFactors, tol=0.000001):

        running_index = None

        for dtag in self.datasets:
            dataset = self.datasets[dtag]
            reflections = dataset.reflections.reflections
            reflections_array = np.array(reflections, copy=True)
            reflections_table = pd.DataFrame(reflections_array,
                                             columns=reflections.column_labels(),
                                             )
            reflections_table.set_index(["H", "K", "L"], inplace=True)

            is_na = reflections_table[structure_factors.f].isna()
            is_zero = reflections_table[structure_factors.f].abs() < tol
            mask = ~(is_na | is_zero)

            flattened_index = reflections_table[mask].index.to_flat_index()
            if running_index is None:
                running_index = flattened_index
            running_index = running_index.intersection(flattened_index)
        return running_index.to_list()

    def truncate(self, resolution: Resolution, structure_factors: StructureFactors) -> Datasets:
        new_datasets_resolution = {}

        # Truncate by common resolution
        for dtag in self.datasets:
            truncated_dataset = self.datasets[dtag].truncate_resolution(resolution, )

            new_datasets_resolution[dtag] = truncated_dataset

        dataset_resolution_truncated = Datasets(new_datasets_resolution)

        # Get common set of reflections
        common_reflections = dataset_resolution_truncated.common_reflections(structure_factors)

        # truncate on reflections
        new_datasets_reflections = {}
        for dtag in dataset_resolution_truncated:
            reflections = dataset_resolution_truncated[dtag].reflections.reflections
            reflections_array = np.array(reflections)

            truncated_dataset = dataset_resolution_truncated[dtag].truncate_reflections(common_reflections,
                                                                                        )
            reflections = truncated_dataset.reflections.reflections
            reflections_array = np.array(reflections)

            new_datasets_reflections[dtag] = truncated_dataset

        return Datasets(new_datasets_reflections)

    def smooth_datasets(self,
                        reference: Reference,
                        structure_factors: StructureFactors,
                        cut=97.5,
                        smooth_func=smooth,
                        mapper=False,
                        ):

        if mapper:
            keys = list(self.datasets.keys())

            results = mapper(
                [
                    # delayed(
                    # self[key].smooth)(
                    # partial(
                        Partial(
                            smooth_func,
                            self[key],
                        reference,
                        structure_factors
                        )
                    for key
                    in keys
                ]
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
               cut=97.5,
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
            knn_x.fit(r.reshape(-1, 1),
                      x.reshape(-1, 1),
                      )
            x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)

            scales = []
            rmsds = []

            for scale in np.linspace(-4, 4, 100):
                y_s = y * np.exp(scale * r)
                knn_y = neighbors.RadiusNeighborsRegressor(0.01)
                knn_y.fit(r.reshape(-1, 1),
                          y_s.reshape(-1, 1),
                          )

                y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

                rmsd = np.sum(np.abs(x_f - y_f))

                scales.append(scale)
                rmsds.append(rmsd)
                print('dtag, scale, rmsd = ',dtag,scale,rmsd)

            min_scale = scales[np.argmin(rmsds)]
            print('min_scale =',min_scale)

            f_array = dtag_reflections_table[structure_factors.f]

            f_scaled_array = f_array * np.exp(min_scale * resolution_array)

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

    # def cluster(self, alignments: Alignments, grid: Grid, reference: Reference, structure_factors: StructureFactors,
    #             mapper=False, sample_rate: float = 3.0) -> Datasets:
    #
    #     # Common res
    #     resolution_high: Resolution = min([self[dtag].reflections.resolution() for dtag in self],
    #                                       key=lambda x: x.resolution,
    #                                       )
    #
    #     print(f"High resolution is: {resolution_high}")
    #
    #     # Truncated datasets
    #     truncated_datasets: Datasets = self.truncate(resolution=resolution_high,
    #                                                  structure_factors=structure_factors,
    #                                                  )
    #
    #     # Truncated reference
    #     truncated_reference_dataset: Dataset = truncated_datasets[reference.dtag]
    #
    #     # Get reference map
    #     reference_xmap: Xmap = Xmap.from_unaligned_dataset_c(
    #         truncated_reference_dataset,
    #         alignments[reference.dtag],
    #         grid,
    #         structure_factors,
    #         sample_rate,
    #     )
    #     # Get reference_array
    #     reference_array: np.ndarray = reference_xmap.to_array()
    #
    #     # Get dtags
    #     keys = list(self.datasets.keys())
    #
    #     # Get correlations
    #     if mapper:
    #         correlations: List[float] = mapper(
    #             delayed(
    #                 self[dtag].correlation)(
    #                 alignments[dtag],
    #                 grid,
    #                 structure_factors,
    #                 reference_array,
    #                 sample_rate,
    #             )
    #             for dtag
    #             in keys
    #         )
    #     else:
    #         correlations: List[float] = [
    #             self[dtag].correlation(
    #                 alignments[dtag],
    #                 grid,
    #                 structure_factors,
    #                 reference_array,
    #                 sample_rate,
    #             )
    #             for dtag
    #             in keys
    #         ]
    #
    #     for i, dtag in enumerate(keys):
    #         print(
    #             (
    #                 f"Dtag: {dtag}; Correlation: {correlations[i]} \n"
    #             )
    #         )
    #
    #     new_datasets = {keys[i]: self.datasets[dtag]
    #                     for i, dtag
    #                     in enumerate(keys)
    #                     if correlations[i] > 0.7
    #                     }
    #
    #     return Datasets(new_datasets)
