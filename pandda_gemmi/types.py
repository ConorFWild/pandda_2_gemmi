from __future__ import annotations

import typing
import dataclasses

import re
from pathlib import Path

import numpy as np
import scipy
from scipy import spatial
from scipy import stats
from scipy.cluster.hierarchy import fclusterdata

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
            print([dtag, dataset_dir])
            dataset: Dataset = Dataset.from_files(dataset_dir.input_pdb_file,
                                                  dataset_dir.input_mtz_file,
                                                  )

            datasets[dtag] = dataset

        return Datasets(datasets)

    def __getitem__(self, item):
        return self.datasets[item]

    def remove_dissimilar_models(self, reference: Reference, max_rmsd_to_reference: float) -> Datasets:
        for dtag in self.datasets:
            print(dtag)
            print(RMSD.from_structures(self.datasets[dtag].structure,
                                       reference.dataset.structure,
                                       ).to_float())

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
    protein_mask: gemmi.Int8Grid
    symmetry_mask: gemmi.Int8Grid

    def __getitem__(self, item: ResidueID):
        return self.partitioning[item]

    @staticmethod
    def from_reference(reference: Reference,
                       grid: gemmi.FloatGrid,
                       mask_radius: float,
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

        mask = gemmi.Int8Grid(*[grid.nu, grid.nv, grid.nw])
        mask.spacegroup = grid.spacegroup
        mask.set_unit_cell(grid.unit_cell)
        for atom in reference.dataset.structure.protein_atoms():
            pos = atom.pos
            mask.set_points_around(pos,
                                   radius=mask_radius,
                                   value=1,
                                   )

        mask_array = np.array(mask, copy=False)

        coord_array = np.argwhere(mask_array == 1)

        query_points = coord_array / spacing

        distances, indexes = kdtree.query(query_points)

        partitions = {}
        for i, coord_as_array in enumerate(coord_array):
            coord = (coord_as_array[0], coord_as_array[1], coord_as_array[2])

            res_num = indexes[i]

            res_id = res_indexes[res_num]

            if res_id not in partitions:
                partitions[res_id] = {}

            partitions[res_id][coord] = grid.unit_cell.orthogonalize(gemmi.Fractional(coord[0] / spacing[0],
                                                                                      coord[1] / spacing[1],
                                                                                      coord[2] / spacing[2],
                                                                                      )
                                                                     )

        symmetry_mask = Partitioning.get_symmetry_contact_mask(reference, mask, mask_radius)

        return Partitioning(partitions, mask, symmetry_mask)

    @staticmethod
    def get_symmetry_contact_mask(reference: Reference, protein_mask: gemmi.Int8Grid,
                                  symmetry_mask_radius: float = 3):
        protein_mask_array = np.array(protein_mask, copy=False)

        mask = gemmi.Int8Grid(*protein_mask_array.shape)
        mask.spacegroup = protein_mask.spacegroup
        mask.set_unit_cell(protein_mask.unit_cell)

        print("\tGetting symops")
        symops = Symops.from_grid(mask)
        print([symmetry_operation for symmetry_operation in symops])

        print("\tIterating")
        for atom in reference.dataset.structure.protein_atoms():
            for symmetry_operation in symops.symops[1:]:
                position = atom.pos
                fractional_position = mask.unit_cell.fractionalize(position)
                symmetry_position = gemmi.Fractional(*symmetry_operation.apply_to_xyz([fractional_position[0],
                                                                                       fractional_position[1],
                                                                                       fractional_position[2],
                                                                                       ]))
                orthogonal_symmetry_position = mask.unit_cell.orthogonalize(symmetry_position)

                mask.set_points_around(orthogonal_symmetry_position,
                                       radius=symmetry_mask_radius,
                                       value=1,
                                       )

        mask_array = np.array(mask, copy=False)
        print("\tGot symmetry mask of size {}, shape {}".format(np.sum(mask_array), mask_array.shape))

        protein_mask_array = np.array(protein_mask, copy=False)

        equal_mask = protein_mask_array == mask_array

        print("\tequal mask of size {}".format(np.sum(equal_mask)))

        mask_array[:, :, :] = mask_array * protein_mask_array

        return mask


@dataclasses.dataclass()
class Grid:
    grid: gemmi.FloatGrid
    partitioning: Partitioning

    @staticmethod
    def from_reference(reference: Reference, mask_radius: float):
        unit_cell = Grid.unit_cell_from_reference(reference)
        spacing: typing.List[int] = Grid.spacing_from_reference(reference)

        grid = gemmi.FloatGrid(*spacing)
        grid.unit_cell = unit_cell
        grid.spacegroup = reference.dataset.reflections.spacegroup()

        partitioning = Partitioning.from_reference(reference,
                                                   grid,
                                                   mask_radius, )

        return Grid(grid, partitioning)

    def new_grid(self):
        spacing = [self.grid.nu, self.grid.nv, self.grid.nw]
        unit_cell = self.grid.unit_cell
        grid = gemmi.FloatGrid(spacing[0], spacing[1], spacing[2])
        grid.unit_cell = unit_cell
        grid.spacegroup = self.grid.spacegroup
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
    com: np.array

    def apply(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
        transform = self.transform
        transformed_positions = {}
        for index, position in positions.items():
            transformed_vector = transform.apply(position)
            transformed_positions[index] = gemmi.Position(transformed_vector[0],
                                                          transformed_vector[1],
                                                          transformed_vector[2],
                                                          )

        return transformed_positions

    def apply_inverse(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
        inverse_transform = self.transform.inverse()
        transformed_positions = {}
        for index, position in positions.items():
            rotation_frame_position = gemmi.Position(position[0] - self.com[0],
                                                     position[1] - self.com[1],
                                                     position[2] - self.com[2])
            transformed_vector = inverse_transform.apply(rotation_frame_position)

            transformed_positions[index] = gemmi.Position(transformed_vector[0] + self.com[0],
                                                          transformed_vector[1] + self.com[1],
                                                          transformed_vector[2] + self.com[2])

        return transformed_positions

    @staticmethod
    def from_translation_rotation(translation, rotation, com):
        transform = gemmi.Transform()
        transform.vec.fromlist(translation.tolist())
        transform.mat.fromlist(rotation.as_matrix().tolist())

        return Transform(transform, com)

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

        vec = mean_ref - mean

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com = mean_ref

        return Transform.from_translation_rotation(vec, rotation, com)

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

        vec = mean_ref - mean

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com = mean_ref

        return Transform.from_translation_rotation(vec, rotation, com)

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

        vec = mean_ref - mean

        de_meaned = matrix - mean
        de_meaned_ref = matrix_ref - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com = mean_ref

        return Transform.from_translation_rotation(vec, rotation, com)


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
                                               gemmi.Position] = alignment[residue_id].apply_inverse(
                alignment_positions)

            # transformed_positions_fractional: typing.Dict[typing.Tuple[int], gemmi.Fractional] = {
            #     point: unaligned_xmap.unit_cell.fractionalize(pos) for point, pos in transformed_positions.items()}

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

    def save(self, path: Path):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = self.xmap
        ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(path))


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
    sigma_is: typing.Dict[Dtag, float]
    sigma_s_m: np.ndarray

    @staticmethod
    def from_xmaps(xmaps: Xmaps, grid: Grid):
        mask = grid.partitioning.protein_mask
        mask_array = np.array(mask, copy=False, dtype=np.bool)

        arrays = {}
        for dtag in xmaps:
            xmap = xmaps[dtag]
            xmap_array = xmap.to_array()
            # print(xmap_array.shape)
            # print(mask_array.shape)
            arrays[dtag] = xmap_array[mask_array]

        stacked_arrays = np.stack(list(arrays.values()), axis=0)
        mean_flat = np.mean(stacked_arrays, axis=0)

        # Estimate the dataset residual variability
        sigma_is = {}
        for dtag in xmaps:
            sigma_i = Model.calculate_sigma_i(mean_flat,
                                              arrays[dtag])
            sigma_is[dtag] = sigma_i
            print([dtag, sigma_i])

        # Estimate the adjusted pointwise variance
        sigma_is_array = np.array(list(sigma_is.values()))[:, np.newaxis]
        sigma_s_m_flat = Model.calculate_sigma_s_m(mean_flat,
                                                   stacked_arrays,
                                                   sigma_is_array,
                                                   )

        mean = np.zeros(mask_array.shape)
        mean[mask_array] = mean_flat

        sigma_s_m = np.zeros(mask_array.shape)
        sigma_s_m[mask_array] = sigma_s_m_flat

        return Model(mean,
                     sigma_is,
                     sigma_s_m,
                     )

    @staticmethod
    def calculate_sigma_i(mean: np.array, array: np.array):
        # TODO: Make sure this is actually equivilent
        # Calculated from slope of array - mean distribution against normal(0,1)
        residual = np.subtract(array, mean)
        # sigma_i = np.std(residual)
        # sigma_i = 0.01

        percentiles = np.linspace(0, 1, array.size + 2)[1:-1]
        normal_quantiles = stats.norm.ppf(percentiles)
        observed_quantile_estimates = np.sort(residual)

        above_min_mask = normal_quantiles > -1
        below_max_mask = normal_quantiles < 1
        centre_mask = above_min_mask * below_max_mask

        central_theoretical_quantiles = normal_quantiles[centre_mask]
        central_observed_quantiles = observed_quantile_estimates[centre_mask]

        map_unc, map_off = np.polyfit(x=central_theoretical_quantiles, y=central_observed_quantiles, deg=1)

        return map_unc

    @staticmethod
    def calculate_sigma_s_m(mean: np.array, arrays: np.arrays, sigma_is_array: typing.Dict[Dtag, float]):
        # Maximise liklihood of data at m under normal(mu_m, sigma_i + sigma_s_m) by optimising sigma_s_m
        # mean[m]
        # arrays[n,m]
        # sigma_i_array[n]
        #
        print("\tMean shape: {}".format(mean.shape))
        print("\tarrays shape: {}".format(arrays.shape))
        print("\tsigma_is_array shape: {}".format(sigma_is_array.shape))
        print(arrays[:, 0])
        print(mean[0])
        print(sigma_is_array[0, :])
        print(arrays[:, -1])
        print(mean[-1])
        print(sigma_is_array[-1, :])

        # print("First log liklihood")
        # print(Model.log_liklihood(1e-16, mean[0], arrays[:,0], sigma_is_array[0,0]))
        #

        func = lambda est_sigma: Model.log_liklihood(est_sigma, mean, arrays, sigma_is_array)

        shape = mean.shape
        num = len(sigma_is_array)

        sigma_ms = Model.vectorised_optimisation_bisect(func,
                                                        0,
                                                        10,
                                                        30,
                                                        arrays.shape
                                                        )

        return sigma_ms

    @staticmethod
    def vectorised_optimisation_bf(func, start, stop, num, shape):
        xs = np.linspace(start, stop, num)

        val = np.ones(shape) * xs[0] + 1.0 / 10000000000000000000000.0
        res = np.ones((shape[1], shape[2], shape[3])) * xs[0]

        y_max = func(val)

        for x in xs[1:]:
            print("\tX equals {}".format(x))
            val[:, :, :, :] = 1
            val = val * x

            y = func(x)
            print(np.max(y))

            y_above_y_max_mask = y > y_max
            y_max[y_above_y_max_mask] = y[y_above_y_max_mask]
            res[y_above_y_max_mask] = x
            print("\tUpdated {} ress".format(np.sum(y_above_y_max_mask)))

        return res

    @staticmethod
    def vectorised_optimisation_bisect(func, start, stop, num, shape):
        # Define step 0
        x_lower_orig = (np.ones(shape[1]) * start) + 1e-16

        x_upper_orig = np.ones(shape[1]) * stop

        f_lower = func(x_lower_orig[np.newaxis, :])
        f_upper = func(x_upper_orig[np.newaxis, :])

        test_mat = f_lower * f_upper
        test_mat_mask = test_mat > 0

        print("Number of points that fail bisection is: {}/{}".format(np.sum(test_mat_mask), test_mat_mask.size))
        print("fshape is {}".format(f_upper.shape))

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
        term2 = np.ones(est_sigma.shape) / (np.square(est_sigma) + np.square(obs_error))
        return np.sum(term1, axis=0) - np.sum(term2, axis=0)

    def evaluate(self, xmap: Xmap, dtag: Dtag):
        xmap_array = np.copy(xmap.to_array())

        residuals = (xmap_array - self.mean)
        denominator = (np.sqrt(np.square(self.sigma_s_m) + np.square(self.sigma_is[dtag])))

        return residuals / denominator


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

    def to_array(self):
        return np.array(self.zmap, copy=True)

    def shape(self):
        return [self.zmap.nu, self.zmap.nv, self.zmap.nw]

    def spacegroup(self):
        return self.zmap.spacegroup

    def unit_cell(self):
        return self.zmap.unit_cell

    def save(self, path: Path):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = self.zmap
        ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(path))


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

        protein_mask_grid = Clustering.get_protein_mask(zmap,
                                                        reference,
                                                        masks.outer_mask,
                                                        )
        print("\tGor protein mask")

        protein_mask = np.array(protein_mask_grid, copy=False, dtype=np.bool)

        symmetry_contact_mask_grid = Clustering.get_symmetry_contact_mask(zmap,
                                                                          reference,
                                                                          protein_mask,
                                                                          symmetry_mask_radius=masks.inner_mask,
                                                                          )

        print("\tGot symmetry mask")

        symmetry_contact_mask = np.array(symmetry_contact_mask_grid, copy=False, dtype=np.bool)

        # Don't consider outlying points away from the protein
        zmap_array[~protein_mask] = 0.0

        # Don't consider outlying points at symmetry contacts
        zmap_array[symmetry_contact_mask] = 0.0

        print(np.sum(protein_mask))
        print(np.sum(symmetry_contact_mask))
        print(np.sum(protein_mask * symmetry_contact_mask))

        print("\tMask contour level is: {}".format(masks.contour_level))
        extrema_mask_array = zmap_array > masks.contour_level
        print("\t{} non extra points".format(np.sum(extrema_mask_array.shape)))

        # extrema_mask_array = zmap_array > 1.0
        # print("\t{} non extra points".format(np.sum(extrema_mask_array)))
        #
        # extrema_mask_array = zmap_array > 0.5
        # print("\t{} non extra points".format(np.sum(extrema_mask_array)))

        extrema_grid_coords_array = np.argwhere(extrema_mask_array)  # n,3
        print("\toutlier array shape: {}".format(extrema_grid_coords_array.shape))

        grid_dimensions_array = np.array([zmap.zmap.unit_cell.a,
                                          zmap.zmap.unit_cell.b,
                                          zmap.zmap.unit_cell.c,
                                          ])

        extrema_fractional_coords_array = extrema_grid_coords_array / grid_dimensions_array  # n,3

        positions = []
        for point in extrema_fractional_coords_array:
            position = gemmi.Fractional(*point)
            pos_orth = zmap.zmap.unit_cell.orthogonalize(position)
            pos_orth_array = [pos_orth[0],
                              pos_orth[1],
                              pos_orth[2], ]
            positions.append(pos_orth_array)

        extrema_cart_coords_array = np.array(positions)  # n, 3
        print("\tshape: {}".format(extrema_cart_coords_array.shape))

        print("\tClustering")
        cluster_ids_array = fclusterdata(X=extrema_cart_coords_array,
                                         t=blob_finding.clustering_cutoff,
                                         criterion='distance',
                                         metric='euclidean',
                                         method='single',
                                         )

        clusters = {}
        for unique_cluster in np.unique(cluster_ids_array):
            cluster_mask = cluster_ids_array == unique_cluster  # n

            cluster_indicies = np.nonzero(cluster_mask)  # (n')

            cluster_points_array = extrema_grid_coords_array[cluster_indicies]

            cluster_points_tuple = (cluster_points_array[:, 0],
                                    cluster_points_array[:, 1],
                                    cluster_points_array[:, 2],)

            # indexes = np.unravel_index(cluster_points_tuple,
            #                            zmap_array.shape,
            #                            )  # (n',n',n')

            values = zmap_array[cluster_points_tuple]

            cluster = Cluster(cluster_points_tuple,
                              values,
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

    @staticmethod
    def get_symmetry_contact_mask(zmap: Zmap, reference: Reference, protein_mask: gemmi.Int8Grid,
                                  symmetry_mask_radius: float = 3):
        mask = gemmi.Int8Grid(*zmap.shape())
        mask.spacegroup = zmap.spacegroup()
        mask.set_unit_cell(zmap.unit_cell())

        print("\tGetting symops")
        symops = Symops.from_grid(mask)
        print([symmetry_operation for symmetry_operation in symops])

        print("\tIterating")
        for atom in reference.dataset.structure.protein_atoms():
            for symmetry_operation in symops.symops[1:]:
                position = atom.pos
                fractional_position = mask.unit_cell.fractionalize(position)
                symmetry_position = gemmi.Fractional(*symmetry_operation.apply_to_xyz([fractional_position[0],
                                                                                       fractional_position[1],
                                                                                       fractional_position[2],
                                                                                       ]))
                orthogonal_symmetry_position = mask.unit_cell.orthogonalize(symmetry_position)

                mask.set_points_around(orthogonal_symmetry_position,
                                       radius=symmetry_mask_radius,
                                       value=1,
                                       )

        mask_array = np.array(mask, copy=False)
        print("\tGot symmetry mask of size {}, shape {}".format(np.sum(mask_array), mask_array.shape))

        protein_mask_array = np.array(protein_mask, copy=False)

        equal_mask = protein_mask_array == mask_array

        print("\tequal mask of size {}".format(np.sum(equal_mask)))

        mask_array[:, :, :] = mask_array * protein_mask_array

        return mask

    def __getitem__(self, item):
        return self.clustering[item]

    def __len__(self):
        return len(self.clustering)


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

        # positions_1 = [ atom.pos for atom in structure_1.protein_atoms()]
        # positions_2 = [ atom.pos for atom in structure_2.protein_atoms()]
        #

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

        # print(positions_1_array)

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
        array_1_mean = np.mean(array_1, axis=0).reshape((1, 3))
        array_2_mean = np.mean(array_2, axis=0).reshape((1, 3))

        # print("#################################")
        # print(array_1)
        # print(array_1_mean)

        print([array_1.shape, array_1_mean.shape])
        array_1_demeaned = array_1 - array_1_mean
        array_2_demeaned = array_2 - array_2_mean
        # print(array_1_demeaned)
        #
        # print(array_1_demeaned-array_2_demeaned)
        print("rmsd is: {}".format(np.sqrt(
            np.sum(np.square(np.linalg.norm(array_1_demeaned - array_2_demeaned, axis=1)), axis=0) / array_1.shape[0])))
        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(array_1_demeaned, array_2_demeaned)
        rotated_vecs = rotation.apply(array_2_demeaned)
        # print("rmsd after rotation is: {}".format(np.sqrt(
        #     np.sum(np.square(np.linalg.norm(array_1_demeaned - rotated_vecs, axis=1)), axis=0) / array_1.shape[0])))
        # print("scipy thinks rmsd is: {}".format(rmsd))
        #
        true_rmsd = np.sqrt(
            np.sum(np.square(np.linalg.norm(array_1_demeaned - rotated_vecs, axis=1)), axis=0) / array_1.shape[0])

        # print(rotation)
        # print(rmsd)

        return RMSD(true_rmsd)

    def to_float(self):
        return self.rmsd
