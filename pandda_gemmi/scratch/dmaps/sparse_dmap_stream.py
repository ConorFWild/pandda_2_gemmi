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

        for residue_id in dframe.partitioning.partitions:
            point_position_array = dframe.partitioning.partitions[residue_id]

            al = alignment.transforms[residue_id]

            transform = al.get_transform()
            com_moving = al.com_moving
            com_reference = al.com_reference

            points = point_position_array.points.tolist()
            positions = point_position_array.positions.tolist()

            gemmi.interpolate_points_single(
                xmap,
                aligned_xmap,
                points,
                positions,
                transform,
                com_moving,
                com_reference,
            )

        return SparseDMap.from_xmap(aligned_xmap, dframe)

    def __getitem__(self, dtag):
        ...

