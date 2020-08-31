from __future__ import annotations

import typing
import dataclasses

import re
from pathlib import Path

import numpy as np
import scipy
from scipy import spatial
import pandas as pd
import gemmi

from pandda_gemmi.constants import *
from pandda_gemmi.config import *


@dataclasses.dataclass()
class Dtag:
    dtag: str

    def __hash__(self):
        return hash(self.dtag)


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

    def __hash__(self):
        return hash((self.model, self.chain, self.insertion))


@dataclasses.dataclass()
class RFree:
    rfree: float

    @staticmethod
    def from_structure(structure: Structure):
        # print(list(structure.structure.make_mmcif_document()[0].find_loop("_refine.ls_R_factor_R_free"))[0])
        rfree = structure.structure.make_mmcif_document()[0].find_loop("_refine.ls_R_factor_R_free")[0]
        # print([[item for item in x] for x in structure.structure.make_mmcif_document()])

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

    @staticmethod
    def from_file(file: Path) -> Structure:
        structure = gemmi.read_structure(str(file))
        return Structure(structure)

    def rfree(self):
        return RFree.from_structure(self)

    def __getitem__(self, item: ResidueID):
        return self.structure[item.model][item.chain][item.insertion]

    def residue_ids(self):
        residue_ids = []
        for model in self.structure:
            for chain in model:
                for residue in chain.get_polymer():
                    resid = ResidueID.from_residue_chain(model, chain, residue)
                    residue_ids.append(resid)

        return residue_ids

    def protein_atoms(self):
        for model in self.structure:
            for chain in model:
                for residue in chain.get_polymer():
                    for atom in residue:
                        yield atom


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

    @staticmethod
    def from_file(file: Path) -> Reflections:
        reflections = gemmi.read_mtz_file(str(file))
        return Reflections(reflections)

    def resolution(self) -> Resolution:
        return Resolution.from_float(self.reflections.resolution_high())

    def truncate(self, resolution: Resolution) -> Reflections:
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = self.reflections.spacegroup
        new_reflections.set_cell_for_all(self.reflections.cell)

        # Add dataset
        new_reflections.add_dataset("truncated")

        # Add columns
        for column in self.reflections.columns:
            new_reflections.add_column(column.label, column.type)

        # Update data
        old_data = np.array(self.reflections, copy=True)
        new_reflections.set_data(old_data[self.reflections.make_d_array() >= resolution.resolution])

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


@dataclasses.dataclass()
class Dataset:
    structure: Structure
    reflections: Reflections

    @staticmethod
    def from_files(pdb_file: Path, mtz_file: Path):
        strucure: Structure = Structure.from_file(pdb_file)
        reflections: Reflections = Reflections.from_file(mtz_file)

        return Dataset(structure=strucure,
                       reflections=reflections,
                       )

    def truncate(self, resolution: Resolution) -> Dataset:
        return Dataset(self.structure,
                       self.reflections.truncate(resolution))


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

    def scale_reflections(self):
        # Scale to the reference dataset
        return self

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

    def truncate(self, resolution: Resolution) -> Datasets:
        new_datasets = {}

        for dtag in self.datasets:
            truncated_dataset = self.datasets[dtag].truncate(resolution)

            new_datasets[dtag] = truncated_dataset

        return Datasets(new_datasets)

    def __iter__(self):
        for dtag in self.datasets:
            yield dtag


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
    partitioning: typing.Dict[ResidueID, typing.Dict[typing.Tuple[int], gemmi.Position]]

    def __getitem__(self, item: ResidueID):
        return self.partitioning[item]

    @staticmethod
    def from_reference(reference: Reference,
                       grid: gemmi.FloatGrid,
                       ):

        array = np.array(grid, copy=False)

        spacing = np.array([grid.nu, grid.nv, grid.nw])

        poss = []
        res_indexes = {}
        i = 0
        for model in reference.dataset.structure.structure:
            for chain in model:
                for res in chain.get_polymer():
                    ca = res["CA"][0]

                    position = ca.pos

                    fractional = grid.unit_cell.fractionalize(position)

                    poss.append(fractional)

                    res_indexes[i] = ResidueID.from_residue_chain(model, chain, res)
                    i = i + 1

        ca_position_array = np.array([[x for x in pos] for pos in poss])

        kdtree = spatial.KDTree(ca_position_array)

        coord_array = [coord for coord, val in np.ndenumerate(array)]

        query_points = np.array([[x / spacing[i] for i, x in enumerate(coord)] for coord in coord_array])

        distances, indexes = kdtree.query(query_points)

        partitions = {}
        for i, coord in enumerate(coord_array):
            res_num = indexes[i]

            res_id = res_indexes[res_num]

            if res_id not in partitions:
                partitions[res_id] = {}

            partitions[res_id][coord] = grid.unit_cell.orthogonalize(gemmi.Fractional(coord[0] / spacing[0],
                                                                                      coord[1] / spacing[1],
                                                                                      coord[2] / spacing[2],
                                                                                      )
                                                                     )

        return Partitioning(partitions)


@dataclasses.dataclass()
class Grid:
    grid: gemmi.FloatGrid
    partitioning: Partitioning

    @staticmethod
    def from_reference(reference: Reference):
        unit_cell = Grid.unit_cell_from_reference(reference)
        spacing: typing.List[int] = Grid.spacing_from_reference(reference)

        grid = gemmi.FloatGrid(*spacing)
        grid.unit_cell = unit_cell

        partitioning = Partitioning.from_reference(reference,
                                                   grid,
                                                   )

        return Grid(grid, partitioning)

    def new_grid(self):
        spacing = [self.grid.nu, self.grid.nv, self.grid.nw]
        unit_cell = self.grid.unit_cell
        grid = gemmi.FloatGrid(spacing[0], spacing[1], spacing[2])
        grid.unit_cell = unit_cell
        return grid

    @staticmethod
    def spacing_from_reference(reference: Reference):
        spacing = reference.dataset.reflections.reflections.get_size_for_hkl()
        return spacing

    @staticmethod
    def unit_cell_from_reference(reference: Reference):
        return reference.dataset.reflections.reflections.cell

    def __getitem__(self, item):
        return self.grid[item]

    def volume(self) -> float:
        unit_cell = self.grid.unit_cell
        return unit_cell.a * unit_cell.b * unit_cell.c

    def size(self) -> int:
        grid = self.grid
        return grid.nu * grid.nv * grid.nw


@dataclasses.dataclass()
class Transform:
    transform: gemmi.Transform

    def apply(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
        transformed_positions = {}
        for index, position in positions.items():
            transformed_vector = self.transform.apply(position)
            transformed_positions[index] = gemmi.Position(transformed_vector[0],
                                                          transformed_vector[1],
                                                          transformed_vector[2],
                                                          )

        return transformed_positions

    @staticmethod
    def from_translation_rotation(translation, rotation):
        transform = gemmi.Transform()
        transform.vec.fromlist(translation.tolist())
        transform.mat.fromlist(rotation.as_matrix().tolist())

        return Transform(transform)

    @staticmethod
    def from_residues(previous_res, current_res, next_res, previous_ref, current_ref, next_ref):
        previous_ca_pos = previous_res["CA"][0].pos
        current_ca_pos = current_res["CA"][0].pos
        next_ca_pos = next_res["CA"][0].pos

        previous_ref_ca_pos = previous_ref["CA"][0].pos
        current_ref_ca_pos = current_ref["CA"][0].pos
        next_ref_ca_pos = next_ref["CA"][0].pos

        matrix = np.array([Transform.pos_to_list(previous_ca_pos),
                           Transform.pos_to_list(current_ca_pos),
                           Transform.pos_to_list(next_ca_pos), ])
        matrix_ref = np.array([Transform.pos_to_list(previous_ref_ca_pos),
                               Transform.pos_to_list(current_ref_ca_pos),
                               Transform.pos_to_list(next_ref_ca_pos), ])

        mean = np.mean(matrix, axis=0)
        mean_ref = np.mean(matrix_ref, axis=0)

        vec = mean_ref - mean

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        return Transform.from_translation_rotation(vec, rotation)

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
            Transform.pos_to_list(next_ca_pos), ])
        matrix_ref = np.array([
            Transform.pos_to_list(current_ref_ca_pos),
            Transform.pos_to_list(next_ref_ca_pos), ])

        mean = np.mean(matrix, axis=0)
        mean_ref = np.mean(matrix_ref, axis=0)

        vec = mean_ref - mean

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        return Transform.from_translation_rotation(vec, rotation)

    @staticmethod
    def from_finish_residues(previous_res, current_res, previous_ref, current_ref):
        previous_ca_pos = previous_res["CA"][0].pos
        current_ca_pos = current_res["CA"][0].pos

        previous_ref_ca_pos = previous_ref["CA"][0].pos
        current_ref_ca_pos = current_ref["CA"][0].pos

        matrix = np.array([Transform.pos_to_list(previous_ca_pos),
                           Transform.pos_to_list(current_ca_pos), ])
        matrix_ref = np.array([Transform.pos_to_list(previous_ref_ca_pos),
                               Transform.pos_to_list(current_ref_ca_pos), ])

        mean = np.mean(matrix, axis=0)
        mean_ref = np.mean(matrix_ref, axis=0)

        vec = mean_ref - mean

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        return Transform.from_translation_rotation(vec, rotation)


@dataclasses.dataclass()
class Alignment:
    transforms: typing.Dict[ResidueID, Transform]

    def __getitem__(self, item: ResidueID):
        return self.transforms[item]

    @staticmethod
    def from_dataset(reference: Reference, dataset: Dataset):

        transforms = {}

        for model in dataset.structure.structure:
            for chain in model:
                for res in chain.get_polymer():
                    prev_res = chain.previous_residue(res)
                    next_res = chain.next_residue(res)

                    if prev_res:
                        prev_res_id = ResidueID.from_residue_chain(model, chain, prev_res)
                    current_res_id = ResidueID.from_residue_chain(model, chain, res)
                    if next_res:
                        next_res_id = ResidueID.from_residue_chain(model, chain, next_res)

                    if prev_res:
                        prev_res_ref = reference.dataset.structure[prev_res_id][0]
                    current_res_ref = reference.dataset.structure[current_res_id][0]
                    if next_res:
                        next_res_ref = reference.dataset.structure[next_res_id][0]

                    if not prev_res:
                        transform = Transform.from_start_residues(res, next_res,
                                                                  current_res_ref, next_res_ref)

                    if not next_res:
                        transform = Transform.from_finish_residues(prev_res, res,
                                                                   prev_res_ref, current_res_ref)

                    if prev_res and next_res:
                        transform = Transform.from_residues(prev_res, res, next_res,
                                                            prev_res_ref, current_res_ref, next_res_ref,
                                                            )

                    transforms[current_res_id] = transform

        return Alignment(transforms)

    def __iter__(self):
        for res_id in self.transforms:
            yield res_id


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
    test_dtags: typing.List[Dtag]
    train_dtags: typing.List[Dtag]
    datasets: Datasets
    res_max: Resolution
    res_min: Resolution


@dataclasses.dataclass()
class Shells:
    shells: typing.Dict[int, Shell]

    @staticmethod
    def from_datasets(datasets: Datasets, resolution_binning: ResolutionBinning):

        sorted_dtags = list(sorted(datasets.datasets.keys(),
                                   key=lambda dtag: datasets[dtag].reflections.resolution().resolution,
                                   ))

        train_dtags = sorted_dtags[:resolution_binning.min_characterisation_datasets]

        shells = {}
        shell_num = 0
        shell_dtags = []
        shell_res = datasets[sorted_dtags[0]].reflections.resolution().resolution
        for dtag in sorted_dtags:
            res = datasets[dtag].reflections.resolution().resolution

            if (len(shell_dtags) > resolution_binning.max_shell_datasets) or (
                    res - shell_res > resolution_binning.high_res_increment):
                shell = Shell(shell_dtags,
                              train_dtags,
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
    def from_aligned_dataset(dataset: Dataset, alignment: Alignment, grid: Grid, structure_factors: StructureFactors):
        unaligned_xmap: gemmi.FloatGrid = dataset.reflections.reflections.transform_f_phi_to_map(structure_factors.f,
                                                                                                 structure_factors.phi,
                                                                                                 )

        interpolated_values_tuple = ([], [], [], [])

        for residue_id in alignment:
            alignment_positions: typing.Dict[typing.Tuple[int], gemmi.Position] = grid.partitioning[residue_id]

            transformed_positions: typing.Dict[typing.Tuple[int],
                                               gemmi.Position] = alignment[residue_id].apply(
                alignment_positions)

            interpolated_values: typing.Dict[typing.Tuple[int],
                                             float] = Xmap.interpolate_grid(unaligned_xmap,
                                                                            transformed_positions)

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
    def interpolate_grid(grid: gemmi.FloatGrid, positions: typing.Dict[typing.Tuple[int],
                                                                       gemmi.Position]) -> typing.Dict[
        typing.Tuple[int],
        float]:
        return {coord: grid.interpolate_value(pos) for coord, pos in positions.items()}

    def to_array(self, copy=True):
        return np.array(self.xmap, copy=copy)


@dataclasses.dataclass()
class Xmaps:
    xmaps: typing.Dict[Dtag, Xmap]

    @staticmethod
    def from_datasets(datasets: Datasets):
        pass

    @staticmethod
    def from_aligned_datasets(datasets: Datasets, alignments: Alignments, grid: Grid,
                              structure_factors: StructureFactors):
        xmaps = {}
        for dtag in datasets:
            xmap = Xmap.from_aligned_dataset(datasets[dtag],
                                             alignments[dtag],
                                             grid,
                                             structure_factors)

            xmaps[dtag] = xmap

        return Xmaps(xmaps)

    def __len__(self):
        return len(self.xmaps)

    def __getitem__(self, item):
        return self.xmaps[item]

    def __iter__(self):
        for dtag in self.xmaps:
            yield dtag


@dataclasses.dataclass()
class Model:
    mean: np.array
    std: np.array
    stds: typing.Dict[Dtag, float]

    @staticmethod
    def from_xmaps(xmaps: Xmaps):
        arrays = {}
        for dtag in xmaps:
            xmap = xmaps[dtag]
            xmap_array = xmap.to_array()
            arrays[dtag] = xmap_array

        stacked_arrays = np.stack(list(arrays.values()), axis=-1)
        mean = np.mean(stacked_arrays, axis=-1)

        # Estimate the dataset residual variability
        sigma_is = {}
        for dtag in xmaps:
            sigma_i = Model.calculate_sigma_i(mean, arrays[dtag])
            sigma_is[dtag] = sigma_i

        # Estimate the adjusted pointwise variance
        sigma_s_m = Model.calculate_sigma_s_m(mean, stacked_arrays, sigma_is)

        return Model(mean,
                     sigma_is,
                     sigma_s_m,
                     )

    @staticmethod
    def calculate_sigma_i(mean: np.array, array: np.array):
        # TODO: Make sure this is actually equivilent
        # Calculated from slope of array - mean distribution against normal(0,1)
        residual = array - mean
        sigma_i = np.std(residual)
        return sigma_i

    @staticmethod
    def calculate_sigma_s_m(mean: np.array, arrays: np.arrays, sigma_is: typing.Dict[Dtag, float]):
        # Maximise liklihood of data at m under normal(mu_m, sigma_i + sigma_s_m) by optimising sigma_s_m

        sigma_i_array = np.array(list(sigma_is.values()))
        func = lambda est_sigma: Model.log_liklihood(est_sigma, mean, arrays, sigma_i_array)
        sigma_is = Model.vectorised_optimisation_bf(func,
                                                    0,
                                                    10,
                                                    1000,
                                                    )

        return sigma_is

    @staticmethod
    def vectorised_optimisation_bf(func, start, stop, num):
        xs = np.linspace(start, stop, num)

        y_max = func(xs[0])

        for x in xs[1:]:
            y = func(x)
            y_above_y_max_mask = y > y_max
            y_max[y_above_y_max_mask] = y[y_above_y_max_mask]

        return y_max

    @staticmethod
    def log_liklihood(est_sigma, est_mu, obs_vals, obs_error):
        """Calculate the value of the differentiated log likelihood for the values of mu, sigma"""
        term1 = (obs_vals - est_mu) ** 2 / ((est_sigma ** 2 + obs_error ** 2) ** 2)
        term2 = 1 / (est_sigma ** 2 + obs_error ** 2)
        return np.sum(term1, axis=0) - np.sum(term2, axis=0)

    def evaluate(self, xmap: Xmap, dtag: Dtag):
        xmap_array = np.copy(xmap.to_array())

        xmap_array = (xmap_array - self.mean) / (np.srt(np.square(self.std) + np.square(self.stds[dtag])))

        return xmap_array


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
        new_grid.unit_cell = xmap.xmap.unit_cell

        new_grid_array = np.array(new_grid, copy=True)
        new_grid_array[:, :, :] = zmap_array[:, :, :]

        return new_grid

    def to_array(self):
        return np.array(self.zmap, copy=True)

    def shape(self):
        return [self.zmap.nu, self.zmap.nv, self.zmap.nw]

    def spacegroup(self):
        return self.zmap.spacegroup

    def unit_cell(self):
        return self.zmap.unit_cell


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
        xmap = Xmap.from_aligned_dataset(reference.dataset,
                                         alignment,
                                         grid,
                                         structure_factors,
                                         )

        return ReferenceMap(reference.dtag,
                            xmap)


@dataclasses.dataclass()
class ClusterID:
    dtag: Dtag
    number: int


@dataclasses.dataclass()
class Cluster:
    indexes: typing.Tuple[np.ndarray]
    values: np.ndarray

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
    def from_zmap(zmap: Zmap, reference: Reference, blob_finding: BlobFinding, masks: Masks):
        zmap_array = zmap.to_array()

        protein_mask = Clustering.get_protein_mask(zmap,
                                                   reference,
                                                   masks.inner_mask,
                                                   )
        symmetry_contact_mask = Clustering.get_symmetry_contact_mask(zmap,
                                                                     reference,
                                                                     protein_mask,
                                                                     symmetry_mask_radius=masks.inner_mask_symmetry,
                                                                     )

        # Don't consider outlying points away from the protein
        zmap_array[~protein_mask] = 0

        # Don't consider outlying points at symmetry contacts
        zmap_array[symmetry_contact_mask] = 0

        extrema_mask_array = zmap_array > masks.contour_level
        extrema_grid_coords_array = np.argwhere(extrema_mask_array)

        grid_dimensions_array = np.array(zmap.zmap.unit_cell.a,
                                         zmap.zmap.unit_cell.b,
                                         zmap.zmap.unit_cell.c,
                                         )

        extrema_fractional_coords_array = extrema_grid_coords_array / grid_dimensions_array

        transform_array = zmap.zmap.unit_cell.orthogonalization_matrix

        extrema_cart_coords_array = np.matmul(transform_array, extrema_fractional_coords_array)

        cluster_ids_array = scipy.cluster.hierarchy.fclusterdata(X=extrema_cart_coords_array,
                                                                 t=blob_finding.clustering_cutoff,
                                                                 criterion='distance',
                                                                 metric='euclidean',
                                                                 method='single',
                                                                 )

        clusters = {}
        for unique_cluster in np.unique(cluster_ids_array):
            cluster_mask = cluster_ids_array == unique_cluster

            cluster_indicies = np.nonzero(cluster_mask)

            indexes = np.unravel_index(cluster_indicies,
                                       zmap_array.shape,
                                       )

            values = zmap_array[indexes]

            cluster = Cluster(indexes,
                              values,
                              )
            clusters[unique_cluster] = cluster

        return Clustering(clusters)

    def __iter__(self):
        for cluster_num in self.clustering:
            yield cluster_num

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

        return mask_array

    @staticmethod
    def get_symmetry_contact_mask(zmap: Zmap, reference: Reference, protein_mask: np.array,
                                  symmetry_mask_radius: float = 3):
        mask = gemmi.Int8Grid(*zmap.shape())
        mask.spacegroup = zmap.spacegroup()
        mask.set_unit_cell(zmap.unit_cell())

        symops = Symops.from_grid(mask)

        for atom in reference.dataset.structure.protein_atoms():
            for symmetry_operation in symops:
                position = atom.pos
                fractional_position = mask.unit_cell.fractionalize(position)
                symmetry_position = symmetry_operation.apply_to_xyz(fractional_position)
                orthogonal_symmetry_position = mask.unit_cell.orthogonalize(symmetry_position)

                mask.set_points_around(orthogonal_symmetry_position,
                                       radius=symmetry_mask_radius,
                                       value=1,
                                       )

        mask_array = np.array(mask, copy=False)

        mask_array = mask_array * protein_mask

        return mask_array

    def __getitem__(self, item):
        return self.clustering[item]


@dataclasses.dataclass()
class Clusterings:
    clusters: typing.Dict[Dtag, Clustering]

    @staticmethod
    def from_Zmaps(zmaps: Zmaps, reference: Reference, blob_finding: BlobFinding, masks: Masks):
        clusterings = {}
        for dtag in zmaps:
            clustering = Clustering.from_zmap(zmaps[dtag], reference, blob_finding, masks)
            clusterings[dtag] = clustering

        return Clusterings(clusterings)

    def filter_size(self, grid: Grid, min_cluster_size: float):
        new_clusterings = {}
        for dtag in self.clusters:
            clustering = self.clusters[dtag]
            new_clusters = list(filter(lambda cluster_num: clustering[cluster_num].size() > min_cluster_size,
                                       clustering,
                                       )
                                )

            if len(new_clusters) == 0:
                continue

            else:
                new_clusters_dict = {i: cluster for i, cluster in enumerate(new_clusters)}
                new_clustering = Clustering(new_clusters_dict)
                new_clusterings[dtag] = new_clustering

        return Clusterings(new_clusterings)

    def filter_peak(self, grid: Grid, z_peak: float):
        new_clusterings = {}
        for dtag in self.clusters:
            clustering = self.clusters[dtag]
            new_clusters = list(filter(lambda cluster_num: clustering[cluster_num].peak() > z_peak,
                                       clustering,
                                       )
                                )

            if len(new_clusters) == 0:
                continue

            else:
                new_clusters_dict = {i: cluster for i, cluster in enumerate(new_clusters)}
                new_clustering = Clustering(new_clusters_dict)
                new_clusterings[dtag] = new_clustering

        return Clusterings(new_clusterings)

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


@dataclasses.dataclass()
class BDC:
    bdc: float

    @staticmethod
    def from_float(bdc: float):
        pass

    @staticmethod
    def from_cluster(xmap: Xmap, cluster: Clustering):
        pass


@dataclasses.dataclass()
class Euclidean3Coord:
    x: float
    y: float
    z: float


@dataclasses.dataclass()
class Site:
    Coordinates: Euclidean3Coord


@dataclasses.dataclass()
class Sites:
    sites: typing.Dict[int, Site]

    @staticmethod
    def from_clusters(clusterings: Clusterings):
        pass


@dataclasses.dataclass()
class Event:
    Site: Site
    Bdc: BDC
    Cluster: Cluster
    Coordinate: Euclidean3Coord


@dataclasses.dataclass()
class EventID:
    Dtag: Dtag
    Event_idx: EventIDX


@dataclasses.dataclass()
class Events:
    events: typing.Dict[EventID, Event]

    @staticmethod
    def from_clusters(clusters: Clusterings):
        pass


@dataclasses.dataclass()
class ZMapFile:

    @staticmethod
    def from_zmap(zmap: Zmap):
        pass


@dataclasses.dataclass()
class ZMapFiles:

    @staticmethod
    def from_zmaps(zmaps: Zmaps):
        pass


@dataclasses.dataclass()
class EventMapFile:

    @staticmethod
    def from_event(event: Event, xmap: Xmap):
        pass


@dataclasses.dataclass()
class EventMapFiles:

    @staticmethod
    def from_events(events: Events, xmaps: Xmaps):
        pass


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

    @staticmethod
    def from_pandda_dir(pandda_dir: Path):
        analyses_dir = pandda_dir / PANDDA_ANALYSES_DIR
        pandda_analyse_events_file = pandda_dir / PANDDA_ANALYSE_EVENTS_FILE

        return Analyses(analyses_dir=analyses_dir,
                        pandda_analyse_events_file=pandda_analyse_events_file,
                        )


@dataclasses.dataclass()
class DatasetModels:
    path: Path

    @staticmethod
    def from_dir(path: Path):
        return DatasetModels(path=path)


@dataclasses.dataclass()
class LigandDir:
    pdbs: typing.List[Path]
    cifs: typing.List[Path]
    smiles: typing.List[Path]

    @staticmethod
    def from_path(path):
        pdbs = list(path.glob("*.pdb"))
        cifs = list(path.glob("*.cifs"))
        smiles = list(path.glob("*.smiles"))

        return LigandDir(pdbs,
                         cifs,
                         smiles,
                         )


@dataclasses.dataclass()
class DatasetDir:
    input_pdb_file: Path
    input_mtz_file: Path
    ligand_dir: LigandDir

    @staticmethod
    def from_path(path: Path, input_settings: Input):
        input_pdb_file: Path = next(path.glob(input_settings.pdb_regex))
        input_mtz_file: Path = next(path.glob(input_settings.mtz_regex))
        ligand_dir: LigandDir = LigandDir.from_path(path / PANDDA_LIGAND_FILES_DIR)

        return DatasetDir(input_pdb_file=input_pdb_file,
                          input_mtz_file=input_mtz_file,
                          ligand_dir=ligand_dir,
                          )


@dataclasses.dataclass()
class DataDirs:
    dataset_dirs: typing.Dict[Dtag, DatasetDir]

    @staticmethod
    def from_dir(directory: Path, input_settings: Input):
        dataset_dir_paths = list(directory.glob("*"))

        dataset_dirs = {}

        for dataset_dir_path in dataset_dir_paths:
            dtag = Dtag(dataset_dir_path.name)
            dataset_dir = DatasetDir.from_path(dataset_dir_path, input_settings)
            dataset_dirs[dtag] = dataset_dir

        return DataDirs(dataset_dirs)

    def to_dict(self):
        return self.dataset_dirs


@dataclasses.dataclass()
class ProcessedDataset:
    path: Path
    dataset_models: DatasetModels
    input_mtz: Path
    input_pdb: Path

    @staticmethod
    def from_dataset_dir(dataset_dir: DatasetDir, processed_dataset_dir: Path) -> ProcessedDataset:
        dataset_models_dir = processed_dataset_dir / PANDDA_MODELLED_STRUCTURES_DIR
        input_mtz = dataset_dir.input_mtz_file
        input_pdb = dataset_dir.input_pdb_file

        return ProcessedDataset(path=processed_dataset_dir,
                                dataset_models=DatasetModels.from_dir(dataset_models_dir),
                                input_mtz=input_mtz,
                                input_pdb=input_pdb,
                                )


@dataclasses.dataclass()
class ProcessedDatasets:
    processed_datasets: typing.Dict[Dtag, ProcessedDataset]

    @staticmethod
    def from_data_dirs(data_dirs: DataDirs, processed_datasets_dir: Path):
        processed_datasets = {}
        for dtag, dataset_dir in data_dirs.dataset_dirs.items():
            processed_datasets[dtag] = ProcessedDataset.from_dataset_dir(dataset_dir,
                                                                         processed_datasets_dir / dtag.dtag,
                                                                         )

        return ProcessedDatasets(processed_datasets)


@dataclasses.dataclass()
class PanDDAFSModel:
    pandda_dir: Path
    data_dirs: DataDirs
    analyses: Analyses
    processed_datasets: ProcessedDatasets

    @staticmethod
    def from_dir(input_data_dirs: Path,
                 output_out_dir: Path,
                 input_settings: Input,
                 ):
        analyses = Analyses.from_pandda_dir(output_out_dir)
        data_dirs = DataDirs.from_dir(input_data_dirs, input_settings)
        processed_datasets = ProcessedDatasets.from_data_dirs(data_dirs,
                                                              output_out_dir / PANDDA_PROCESSED_DATASETS_DIR,
                                                              )

        return PanDDAFSModel(pandda_dir=output_out_dir,
                             data_dirs=data_dirs,
                             analyses=analyses,
                             processed_datasets=processed_datasets,
                             )


@dataclasses.dataclass()
class RMSD:
    rmsd: float

    @staticmethod
    def from_structures(structure_1: Structure, structure_2: Structure):
        distances = []

        positions_1 = []
        positions_2 = []

        for residues_id in structure_1.residue_ids():
            res_1 = structure_1[residues_id][0]
            res_2 = structure_2[residues_id][0]

            res_1_ca = res_1["CA"][0]
            res_2_ca = res_2["CA"][0]

            res_1_ca_pos = res_1_ca.pos
            res_2_ca_pos = res_2_ca.pos
            # print(res_1_ca_pos)
            # print(res_2_ca_pos)

            positions_1.append(res_1_ca_pos)
            positions_2.append(res_2_ca_pos)

            distances.append(res_1_ca_pos.dist(res_2_ca_pos))

        positions_1_array = np.array([[x[0], x[1], x[2]] for x in positions_1])
        positions_2_array = np.array([[x[0], x[1], x[2]] for x in positions_2])

        return RMSD.from_arrays(positions_1_array, positions_2_array)
        #
        # distances_array = np.array(distances)
        # print(distances_array)
        # print((1.0 / distances_array.size) )
        # rmsd = np.sqrt((1.0 / distances_array.size) * np.sum(np.square(distances_array)))
        #
        # print(rmsd)
        #
        # return RMSD(rmsd)

    @staticmethod
    def from_arrays(array_1, array_2):
        array_1_mean = np.mean(array_1, axis=0)
        array_2_mean = np.mean(array_2, axis=0)

        # print("#################################")
        # print(array_1)
        # print(array_1_mean)

        array_1_demeaned = array_1 - array_1_mean
        array_2_demeaned = array_2 - array_2_mean
        # print(array_1_demeaned)
        #
        # print(array_1_demeaned-array_2_demeaned)

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(array_1_demeaned, array_2_demeaned)

        # print(rotation)
        # print(rmsd)

        return RMSD(rmsd)

    def to_float(self):
        return self.rmsd
