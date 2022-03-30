from __future__ import annotations

import typing
import dataclasses
import itertools
from pathlib import Path
from typing import Tuple

from joblib.externals.loky import set_loky_pickler

from pandda_gemmi.analyse_interface import LoadXMapInterface
set_loky_pickler('pickle')
import ray

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.python_types import *
from pandda_gemmi.common import Dtag, delayed
from pandda_gemmi.dataset import StructureFactors, Reflections, Dataset, Datasets
from pandda_gemmi.edalignment.alignments import Alignment, Alignments, Transform
from pandda_gemmi.edalignment.grid import Grid, Partitioning


@dataclasses.dataclass()
class Xmap:
    xmap: gemmi.FloatGrid

    @staticmethod
    def from_reflections(reflections: Reflections):
        pass

    @staticmethod
    def from_file(file):
        ccp4 = gemmi.read_ccp4_map(str(file))
        ccp4.setup()
        return Xmap(ccp4.grid)

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

        # for point, position, transform, com_moving, com_reference in zip(point_list, position_list, transform_list, com_moving_list, com_reference_list):

        # print((
        #     f"point: {point}\n"
        #     f"Position: {position}\n"
        #     f"transform: {transform.vec.tolist()} # {transform.mat.tolist()} \n"
        #     f"com moving: {com_moving} \n"
        #     f"Com moving: {com_reference}\n"
        # ))

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

        # for point, position, transform, com_moving, com_reference in zip(point_list, position_list, transform_list, com_moving_list, com_reference_list):

        # print((
        #     f"point: {point}\n"
        #     f"Position: {position}\n"
        #     f"transform: {transform.vec.tolist()} # {transform.mat.tolist()} \n"
        #     f"com moving: {com_moving} \n"
        #     f"Com moving: {com_reference}\n"
        # ))

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
        grid.set_unit_cell(unit_cell)
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
        com_moving = [0.0, 0.0, 0.0]
        com_reference = [0.0, 0.0, 0.0]
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
    def from_aligned_map_c(
            event_map_reference_grid: gemmi.FloatGrid,
            dataset: Dataset,
            alignment: Alignment,
            grid: Grid,
            structure_factors: StructureFactors,
            mask_radius: float,
            partitioning: Partitioning,
            mask_radius_symmetry: float,
            sample_rate: float,
    ):

        moving_xmap_grid: gemmi.FloatGrid = dataset.reflections.reflections.transform_f_phi_to_map(
            structure_factors.f,
            structure_factors.phi,
            # sample_rate=sample_rate,
            sample_rate=dataset.reflections.resolution().resolution/0.5,
        )

        new_grid = gemmi.FloatGrid(*[moving_xmap_grid.nu,
                                     moving_xmap_grid.nv,
                                     moving_xmap_grid.nw])
        new_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        new_grid.set_unit_cell(moving_xmap_grid.unit_cell)

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

    def save(self, path: Path, p1: bool = True):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = self.xmap
        if p1:
            ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        else:
            ccp4.grid.symmetrize_max()
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(path))

    @staticmethod
    def from_grid_array(grid: Grid, array_flat):
        new_grid = grid.new_grid()

        mask = grid.partitioning.protein_mask
        mask_array = np.array(mask, copy=False, dtype=np.int8)

        array = np.zeros(mask_array.shape, dtype=np.float32)
        array[np.nonzero(mask_array)] = array_flat

        for point in new_grid:
            u = point.u
            v = point.v
            w = point.w
            new_grid.set_value(u, v, w, float(array[u, v, w]))

        return Xmap(new_grid)

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

        protein_mask = grid.partitioning.protein_mask
        protein_mask_array = np.array(protein_mask, copy=False, dtype=np.int8)

        symmetry_contact_mask = grid.partitioning.symmetry_mask
        symmetry_contact_mask_array = np.array(symmetry_contact_mask, copy=False, dtype=np.int8)

        arrays = {}
        for dtag in xmaps:
            xmap = xmaps[dtag]
            xmap_array = xmap.to_array()

            array = xmap_array[grid.partitioning.total_mask == 1]

            arrays[dtag] = array

        dtag_list = list(arrays.keys())
        xmap_array = np.stack(list(arrays.values()), axis=0)

        return XmapArray(dtag_list, xmap_array)

    def from_dtags(self, dtags: typing.List[Dtag]):
        bool_mask = []

        for dtag in dtags:
            if dtag not in self.dtag_list:
                raise Exception(f"Dtag {dtag} not in dtags: {self.dtag_list}")

        for dtag in self.dtag_list:
            if dtag in dtags:
                bool_mask.append(True)
            else:
                bool_mask.append(False)

        mask_array = np.array(bool_mask)

        view = self.xmap_array[mask_array]

        return XmapArray([_dtag for _dtag in self.dtag_list if _dtag in dtags], view)

def from_unaligned_dataset_c(dataset: Dataset,
                                  alignment: Alignment,
                                  grid: Grid,
                                  structure_factors: StructureFactors,
                                  sample_rate: float = 3.0, ):
    xmap = Xmap.from_unaligned_dataset_c(dataset,
                                         alignment,
                                         grid,
                                         structure_factors,
                                         # sample_rate,
                                         dataset.reflections.resolution().resolution/0.5
                                         )

    return xmap

class LoadXmap(LoadXMapInterface):
    def __call__(
        self, 
        dataset: DatasetInterface, 
        alignment: AlignmentInterface, 
        grid: GridInterface, 
        structure_factors: StructureFactorsInterface, 
        sample_rate: float = 3) -> XmapInterface:
        return from_unaligned_dataset_c(dataset, alignment, grid, structure_factors, sample_rate)


def from_unaligned_dataset_c_flat(dataset: Dataset,
                                  alignment: Alignment,
                                  grid: Grid,
                                  structure_factors: StructureFactors,
                                  sample_rate: float = 3.0, ):
    xmap = Xmap.from_unaligned_dataset_c(dataset,
                                         alignment,
                                         grid,
                                         structure_factors,
                                         # sample_rate,
                                         dataset.reflections.resolution().resolution/0.5
                                         )

    xmap_array = xmap.to_array()

    masked_array = xmap_array[grid.partitioning.total_mask == 1]

    return masked_array

class LoadXmapFlat(LoadXMapFlatInterface):
    def __call__(
        self, 
        dataset: DatasetInterface, 
        alignment: AlignmentInterface, 
        grid: GridInterface, 
        structure_factors: StructureFactorsInterface, 
        sample_rate: float = 3) -> XmapInterface:
        return from_unaligned_dataset_c_flat(dataset, alignment, grid, structure_factors, sample_rate)


@ray.remote
def from_unaligned_dataset_c_ray(dataset: Dataset,
                                  alignment: Alignment,
                                  grid: Grid,
                                  structure_factors: StructureFactors,
                                  sample_rate: float = 3.0, ):
    xmap = Xmap.from_unaligned_dataset_c(dataset,
                                         alignment,
                                         grid,
                                         structure_factors,
                                         # sample_rate,
                                         dataset.reflections.resolution().resolution/0.5
                                         )

    return xmap


@ray.remote
def from_unaligned_dataset_c_flat_ray(dataset: Dataset,
                                  alignment: Alignment,
                                  grid: Grid,
                                  structure_factors: StructureFactors,
                                  sample_rate: float = 3.0, ):
    xmap = Xmap.from_unaligned_dataset_c(dataset,
                                         alignment,
                                         grid,
                                         structure_factors,
                                         # sample_rate,
                                         dataset.reflections.resolution().resolution/0.5
                                         )

    xmap_array = xmap.to_array()

    masked_array = xmap_array[grid.partitioning.total_mask == 1]

    return masked_array