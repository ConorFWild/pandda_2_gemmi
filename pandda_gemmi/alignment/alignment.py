import dataclasses
import time

import numpy as np
import scipy
from scipy import spatial
import gemmi

from ..interfaces import *
from ..dataset import contains, ResidueID


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

        vec = np.array([0.0, 0.0, 0.0])

        de_meaned_mov = moving_selection - mean_mov
        de_meaned_ref = reference_selection - mean_ref

        rotation, rmsd = spatial.transform.Rotation.align_vectors(de_meaned_mov, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean_mov

        return Transform.from_translation_rotation(vec, rotation, com_reference=com_reference, com_moving=com_moving)


class Alignment:

    def __init__(self,
                 resid, vec, mat, com_reference, com_mov):
        self.resid = resid
        self.vec = vec
        self.mat = mat
        self.com_reference = com_reference
        self.com_mov = com_mov

    def get_transforms(self):
        transforms = {}
        com_ref = {}
        com_mov = {}


        for _j in range(self.resid.shape[0]):

            residue_id = ResidueID(*self.resid[_j])
            transform = gemmi.Transform()
            transform.vec.fromlist(self.vec[_j].tolist())
            transform.mat.fromlist(self.mat[_j].tolist())
            transforms[residue_id] = transform

            com_ref[residue_id] = self.com_reference[_j]
            com_mov[residue_id] = self.com_mov[_j]

        return transforms, com_ref, com_mov

    @classmethod
    def from_structure_arrays(
            cls,
            _dtag,
            moving_structure: StructureArrayInterface,
            reference_structure: StructureArrayInterface,
            marker_atom_search_radius=10.0,
    ):

        #
        used_insertions = []
        ca_mask = []
        for j, atom_id in enumerate(moving_structure.atom_ids):
            key = (moving_structure.chains[j], moving_structure.seq_ids[j])
            if (key not in used_insertions) and contains(str(atom_id).upper(), "CA"):
                ca_mask.append(True)
                used_insertions.append(key)
            else:
                ca_mask.append(False)
        moving_structure_cas = moving_structure.mask(np.array(ca_mask))

        #
        used_insertions = []
        ca_mask = []
        for j, atom_id in enumerate(reference_structure.atom_ids):
            key = (reference_structure.chains[j], reference_structure.seq_ids[j])
            if (key not in used_insertions) and contains(str(atom_id).upper(), "CA"):
                ca_mask.append(True)
                used_insertions.append(key)
            else:
                ca_mask.append(False)
        reference_structure_cas = reference_structure.mask(np.array(ca_mask))

        # Iterate protein atoms, then pull out their atoms, and search them
        ref_ids= np.hstack([
            reference_structure_cas.models.reshape((-1,1)),
            reference_structure_cas.chains.reshape((-1,1)),
            reference_structure_cas.seq_ids.reshape((-1,1))
        ])
        mov_ids = np.hstack([
            moving_structure_cas.models.reshape((-1,1)),
            moving_structure_cas.chains.reshape((-1,1)),
            moving_structure_cas.seq_ids.reshape((-1,1))
        ])
        ids=np.concatenate([ref_ids, mov_ids])

        unique, indicies, counts = np.unique(ids, return_inverse=True, return_counts=True, axis=0)
        count_array = counts[indicies]
        count_mask = count_array > 1  # Mask of ids by count > 1

        ref_pos_mask = count_mask[:ref_ids.shape[0]]
        mov_pos_mask = count_mask[ref_ids.shape[0]:]

        reference_atom_array = reference_structure_cas.positions[ref_pos_mask]
        moving_atom_array = moving_structure_cas.positions[mov_pos_mask]


        if (reference_atom_array.shape[0] == 0) or (moving_atom_array.shape[0] == 0):
            raise Exception(f"{_dtag} Reference atom array shape {reference_atom_array.shape} moving atom array {moving_atom_array.shape}")

        # Other kdtree
        reference_tree = spatial.KDTree(reference_atom_array)

        transforms = []
        time_ball_query = 0
        time_super = 0
        # Start searching
        ref_ids_mask = []
        for _j in range(ref_ids.shape[0]):

            # other selection
            time_begin_ball = time.time()
            reference_indexes = reference_tree.query_ball_point(
                reference_structure_cas.positions[_j,:].flatten(),
                marker_atom_search_radius,
            )
            time_finish_ball = time.time()
            time_ball_query = time_ball_query + (time_finish_ball-time_begin_ball)
            reference_selection = reference_atom_array[reference_indexes]
            moving_selection = moving_atom_array[reference_indexes]

            if moving_selection.shape[0] == 0:
                raise Exception(f"{_dtag} Moving selection shape: {moving_selection.shape[0]} Reference selection shape: {reference_selection.shape[0]}")
            else:
                time_begin_super = time.time()
                ref_ids_mask.append(True)
                transforms.append(
                    Transform.from_atoms(
                        moving_selection,
                        reference_selection,
                        com_moving=np.mean(moving_selection, axis=0),
                        com_reference=np.mean(reference_selection, axis=0),
                )
                )
                time_finish_super = time.time()
                time_super = time_super + (time_finish_super-time_begin_super)

        resid = ref_ids
        vec = np.stack([transform.vec for transform in transforms])
        mat = np.stack([transform.mat for transform in transforms])
        com_reference = np.stack([transform.com_reference for transform in transforms])
        com_mov = np.stack([transform.com_moving for transform in transforms])


        return cls(resid, vec, mat, com_reference, com_mov)