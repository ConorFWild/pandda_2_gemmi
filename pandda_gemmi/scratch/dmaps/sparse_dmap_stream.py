from pathlib import Path

import gemmi

from ..interfaces import *
from .sparse_dmap import SparseDMap


class SparseDMapStream:
    def __init__(self,
                 datasets: Dict[str, DatasetInterface],
                 reference_frame: DFrameInterface,
                 alignments: Dict[str, AlignmentInterface],
                 # cache: Path,
                 transforms
                 ):
        self.datasets = datasets
        self.dframe = reference_frame
        self.alignments = alignments
        self.transforms = transforms
        ...

    def load(self, dtag: str):
        dataset = self.datasets[dtag]
        alignment = self.alignments[dtag]

        for transform in self.transforms:
            dataset = transform(dataset)

        xmap = dataset.reflections.transform_f_phi_to_map()

        aligned_xmap = SparseDMapStream.align_xmap(xmap, self.dframe, alignment)

        return aligned_xmap

    @staticmethod
    def align_xmap(xmap: CrystallographicGridInterface, dframe: DFrameInterface, alignment: AlignmentInterface):

        aligned_xmap = dframe.get_grid()

        # for residue_id in dframe.partitioning.partitions:
        #     point_position_array = dframe.partitioning.partitions[residue_id]
        #
        #     al = alignment.transforms[residue_id]
        #
        #     transform = al.get_transform()
        #     com_moving = al.com_moving
        #     com_reference = al.com_reference
        #
        #     points = point_position_array.points.astype(int).tolist()
        #     # print(f"Points length: {len(points)}")
        #     positions = point_position_array.positions.tolist()
        #
        #     gemmi.interpolate_points_single(
        #         xmap,
        #         aligned_xmap,
        #         points,
        #         positions,
        #         transform,
        #         com_moving,
        #         com_reference,
        #     )
        points_list = [dframe.partitioning.partitions[residue_id].points for residue_id in dframe.partitioning.partitions]
        positions_list = [dframe.partitioning.partitions[residue_id].positions for residue_id in dframe.partitioning.partitions]
        transform_list = [alignment.transforms[residue_id].get_transform() for residue_id in dframe.partitioning.partitions]
        com_moving_list = [alignment.transforms[residue_id].com_moving for residue_id in dframe.partitioning.partitions]
        com_reference_list = [alignment.transforms[residue_id].com_reference for residue_id in dframe.partitioning.partitions]


        gemmi.interpolate_points_multiple(
            xmap,
                    aligned_xmap,
                    points_list,
                    positions_list,
                    transform_list,
                    com_moving_list,
                    com_reference_list,
        )

        return SparseDMap.from_xmap(aligned_xmap, dframe)

    def __getitem__(self, dtag):
        ...

