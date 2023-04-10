import dataclasses
import time

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
            moving_structure: StructureInterface,
            reference_structure: StructureInterface,
            marker_atom_search_radius=10.0,
    ):

        time_begin = time.time()
        moving_pos_list = []
        reference_pos_list = []

        # Iterate protein atoms, then pull out their atoms, and search them
        begin_get_span = time.time()
        for res_id in reference_structure.protein_residue_ids():

            # Get the matchable CAs
            try:
                # Get reference residue
                ref_res_span = reference_structure[res_id]
                ref_res = ref_res_span[0]

                # Get corresponding reses
                mov_res_span = moving_structure[res_id]
                mov_res = mov_res_span[0]

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
        finish_get_span = time.time()

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
        time_ball_query = 0
        time_super = 0
        # Start searching
        for res_id in reference_structure.protein_residue_ids():
            # Get reference residue
            ref_res_span = reference_structure[res_id]
            ref_res = ref_res_span[0]

            # Get ca pos in reference model
            reference_ca_pos = ref_res["CA"][0].pos

            # other selection
            time_begin_ball = time.time()
            reference_indexes = reference_tree.query_ball_point(
                [reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z],
                marker_atom_search_radius,
            )
            time_finish_ball = time.time()
            time_ball_query = time_ball_query + (time_finish_ball-time_begin_ball)
            reference_selection = reference_atom_array[reference_indexes]
            moving_selection = moving_atom_array[reference_indexes]

            if moving_selection.shape[0] == 0:
                # raise ExceptionUnmatchedAlignmentMarker(res_id)
                raise Exception()
            time_begin_super = time.time()
            transforms[res_id] = Transform.from_atoms(
                moving_selection,
                reference_selection,
                com_moving=np.mean(moving_selection, axis=0),
                com_reference=np.mean(reference_selection, axis=0),

            )
            time_finish_super = time.time()
            time_super = time_super + (time_finish_super-time_begin_super)

        # self.transforms = transforms
        self.resid = np.stack([[resid.model, resid.chain, resid.number] for resid in transforms.keys()])
        self.vec = np.stack([transform.vec for transform in transforms.values()])
        self.mat = np.stack([transform.mat for transform in transforms.values()])
        self.com_reference = np.stack([transform.com_reference for transform in transforms.values()])
        self.com_mov = np.stack([transform.com_moving for transform in transforms.values()])


        time_finish = time.time()
        print(f"\t\tAligned in: {round(time_finish-time_begin, 2)} of which {round(time_ball_query,2)} in ball query, {round(finish_get_span-begin_get_span,2)} in getting span and {round(time_super, 2)} in superposition")

    @classmethod
    def from_structure_arrays(
            self,
            moving_structure: StructureArrayInterface,
            reference_structure: StructureArrayInterface,
            marker_atom_search_radius=10.0,
    ):

        time_begin = time.time()
        moving_pos_list = []
        reference_pos_list = []

        used_insertions = []
        ca_mask = []

        #
        for j, atom_id in enumerate(moving_structure.atom_ids):
            key = (moving_structure.chains[j], moving_structure.seq_ids[j])
            if (key not in used_insertions) and contains(str(atom_id).upper(), "CA"):
                ca_mask.append(True)
                used_insertions.append(key)
            else:
                ca_mask.append(False)
        moving_structure_cas = moving_structure.mask(np.array(ca_mask))

        #
        for j, atom_id in enumerate(reference_structure.atom_ids):
            key = (reference_structure.chains[j], reference_structure.seq_ids[j])
            if (key not in used_insertions) and contains(str(atom_id).upper(), "CA"):
                ca_mask.append(True)
                used_insertions.append(key)
            else:
                ca_mask.append(False)
        reference_structure_cas = reference_structure.mask(np.array(ca_mask))

        # Iterate protein atoms, then pull out their atoms, and search them
        begin_get_span = time.time()
        # reference_structure_cas
        ref_ids= np.hstack([
            reference_structure_cas.chains.reshape((-1,1)),
            reference_structure_cas.seq_ids.reshape((-1,1)),
            reference_structure_cas.insertions.reshape((-1,1))
        ])
        mov_ids = np.hstack([
            moving_structure_cas.chains.reshape((-1,1)),
            moving_structure_cas.seq_ids.reshape((-1,1)),
            moving_structure_cas.insertions.reshape((-1,1))
        ])
        print(f"\t\t\tReference id shape: {ref_ids.shape} : Moving structure shape: {mov_ids.shape}")
        ids=np.concat([ref_ids, mov_ids])
        unique, indicies, counts = np.unique(ids, return_inverse=True, return_counts=True, )
        count_array = counts[indicies]
        count_mask = count_array > 1  # Mask of ids by count > 1
        print(f"\t\t\tCount array shape: {count_array.shape} : Count mask shape: {count_mask.shape}")

        ref_pos_mask = count_mask[:ref_ids.shape[0]]
        mov_pos_mask = count_mask[ref_ids.shape[0]:]
        print(f"\t\t\tRef pos mask shape: {ref_pos_mask.shape} : Mov pos mask shape: {mov_pos_mask.shape}")

        reference_atom_array = reference_structure.positions[ref_pos_mask]
        moving_atom_array = moving_structure.positions[mov_pos_mask]

        finish_get_span = time.time()

        # moving_atom_array = np.array(moving_pos_list)
        # reference_atom_array = np.array(reference_pos_list)

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
        time_ball_query = 0
        time_super = 0
        # Start searching
        for res_id in reference_structure.protein_residue_ids():
            # Get reference residue
            ref_res_span = reference_structure[res_id]
            ref_res = ref_res_span[0]

            # Get ca pos in reference model
            reference_ca_pos = ref_res["CA"][0].pos

            # other selection
            time_begin_ball = time.time()
            reference_indexes = reference_tree.query_ball_point(
                [reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z],
                marker_atom_search_radius,
            )
            time_finish_ball = time.time()
            time_ball_query = time_ball_query + (time_finish_ball-time_begin_ball)
            reference_selection = reference_atom_array[reference_indexes]
            moving_selection = moving_atom_array[reference_indexes]

            if moving_selection.shape[0] == 0:
                # raise ExceptionUnmatchedAlignmentMarker(res_id)
                raise Exception()
            time_begin_super = time.time()
            transforms[res_id] = Transform.from_atoms(
                moving_selection,
                reference_selection,
                com_moving=np.mean(moving_selection, axis=0),
                com_reference=np.mean(reference_selection, axis=0),

            )
            time_finish_super = time.time()
            time_super = time_super + (time_finish_super-time_begin_super)

        # self.transforms = transforms
        self.resid = np.stack([[resid.model, resid.chain, resid.number] for resid in transforms.keys()])
        self.vec = np.stack([transform.vec for transform in transforms.values()])
        self.mat = np.stack([transform.mat for transform in transforms.values()])
        self.com_reference = np.stack([transform.com_reference for transform in transforms.values()])
        self.com_mov = np.stack([transform.com_moving for transform in transforms.values()])


        time_finish = time.time()
        print(f"\t\tAligned in: {round(time_finish-time_begin, 2)} of which {round(time_ball_query,2)} in ball query, {round(finish_get_span-begin_get_span,2)} in getting span and {round(time_super, 2)} in superposition")
