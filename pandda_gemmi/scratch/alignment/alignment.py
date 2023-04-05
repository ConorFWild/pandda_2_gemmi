import dataclasses

import numpy as np
import scipy
from scipy import spatial
import gemmi

from ..interfaces import *

@dataclasses.dataclass
class TransformPython:
    transform: List

    @staticmethod
    def from_gemmi(transform_gemmi):
        transform_python = transform_gemmi.mat.tolist()
        return TransformPython(transform_python, )

    def to_gemmi(self):
        transform_gemmi = gemmi.Transform()
        transform_gemmi.mat.fromlist(self.transform)
        return transform_gemmi


@dataclasses.dataclass()
class Transform:
    # transform: gemmi.Transform
    vec: np.array
    mat: np.array
    com_reference: np.array
    com_moving: np.array

    def get_transform(self):
        transform = gemmi.Transform()
        transform.vec.fromlist( self.vec.tolist())
        transform.mat.fromlist(self.mat.tolist())
        return transform

    def apply_moving_to_reference(
            self,
            positions: Dict[Tuple[int], gemmi.Position],
    ) -> Dict[Tuple[int], gemmi.Position]:
        transformed_positions = {}

        transform = self.get_transform()

        for index, position in positions.items():
            rotation_frame_position = gemmi.Position(position[0] - self.com_moving[0],
                                                     position[1] - self.com_moving[1],
                                                     position[2] - self.com_moving[2])
            transformed_vector = transform.apply(rotation_frame_position)

            transformed_positions[index] = gemmi.Position(transformed_vector[0] + self.com_reference[0],
                                                          transformed_vector[1] + self.com_reference[1],
                                                          transformed_vector[2] + self.com_reference[2])

        return transformed_positions

    def apply_reference_to_moving(
            self,
            positions: Dict[Tuple[int], gemmi.Position],
    ) -> Dict[
        Tuple[int], gemmi.Position]:
        inverse_transform = self.get_transform().inverse()
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
        # transform = gemmi.Transform()
        # transform.vec.fromlist(translation.tolist())
        # transform.mat.fromlist(rotation.as_matrix().tolist())

        return Transform(translation, rotation.as_matrix(), com_reference, com_moving)

    @staticmethod
    def pos_to_list(pos: gemmi.Position):
        return [pos[0], pos[1], pos[2]]

    @staticmethod
    def from_atoms(moving_selection,
                   reference_selection,
                   com_moving,
                   com_reference,
                   ):

        mean_mov = com_moving
        mean_ref = com_reference

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned_mov = moving_selection - mean_mov
        de_meaned_ref = reference_selection - mean_ref

        rotation, rmsd = spatial.transform.Rotation.align_vectors(de_meaned_mov, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean_mov

        return Transform.from_translation_rotation(vec, rotation, com_reference=com_reference, com_moving=com_moving)

    # def __getstate__(self):
    #     transform_python = TransformPython.from_gemmi(self.transform)
    #     com_reference = self.com_reference
    #     com_moving = self.com_moving
    #     return (transform_python, com_reference, com_moving)
    #
    # def __setstate__(self, data):
    #     transform_gemmi = data[0].to_gemmi()
    #     self.transform = transform_gemmi
    #     self.com_reference = data[1]
    #     self.com_moving = data[2]


class Alignment:
    def __init__(
            self,
            moving_dataset: DatasetInterface,
            reference_dataset: DatasetInterface,
            marker_atom_search_radius=10.0,
    ):

        moving_pos_list = []
        reference_pos_list = []

        # Iterate protein atoms, then pull out their atoms, and search them
        for res_id in reference_dataset.structure.protein_residue_ids():

            # Get the matchable CAs
            try:
                # Get reference residue
                ref_res_span = reference_dataset.structure[res_id]
                ref_res = ref_res_span[0]

                # Get corresponding reses
                mov_res_span = moving_dataset.structure[res_id]
                mov_res = dataset_res_span[0]

                # Get the CAs
                atom_ref = ref_res["CA"][0]
                atom_mov = mov_res["CA"][0]

                # Get the shared atoms
                reference_pos_list.append([atom_ref.pos.x, atom_ref.pos.y, atom_ref.pos.z, ])
                moving_pos_list.append([atom_mov.pos.x, atom_mov.pos.y, atom_mov.pos.z, ])

            except Exception as e:
                print(
                    f"WARNING: An exception occured in matching residues for alignment at residue id: {res_id}: {e}")
                continue

        moving_atom_array = np.array(moving_pos_list)
        reference_atom_array = np.array(reference_pos_list)

        if (reference_atom_array.shape[0] == 0) or (moving_atom_array.shape[0] == 0):
            # raise ExceptionNoCommonAtoms()
            raise Exception()

        # Other kdtree
        reference_tree = spatial.KDTree(reference_atom_array)

        if reference_atom_array.size != moving_atom_array.size:
            # raise AlignmentUnmatchedAtomsError(reference_atom_array,
            #                                    dataset_atom_array,
            #                                    )
            raise Exception()

        transforms = {}

        # Start searching
        for res_id in reference_dataset.structure.protein_residue_ids():
            # Get reference residue
            ref_res_span = reference_dataset.structure[res_id]
            ref_res = ref_res_span[0]

            # Get ca pos in reference model
            reference_ca_pos = ref_res["CA"][0].pos

            # other selection
            reference_indexes = reference_tree.query_ball_point(
                [reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z],
                marker_atom_search_radius,
            )
            reference_selection = reference_atom_array[reference_indexes]
            moving_selection = moving_atom_array[reference_indexes]

            if moving_selection.shape[0] == 0:
                # raise ExceptionUnmatchedAlignmentMarker(res_id)
                raise Exception()

            transforms[res_id] = Transform.from_atoms(
                moving_selection,
                reference_selection,
                com_moving=np.mean(moving_selection, axis=0),
                com_reference=np.mean(reference_selection, axis=0),

            )

        self.transforms = transforms
