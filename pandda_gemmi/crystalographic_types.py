from __future__ import annotations

from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np

import scipy
from scipy import spatial

import gemmi

@dataclass()
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




@dataclass()
class Transform:
    transform: gemmi.Transform
    com_reference: np.array
    com_moving: np.array

    def apply_moving_to_reference(self, positions: Dict[Tuple[int], gemmi.Position]) -> Dict[
        Tuple[int], gemmi.Position]:
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

    def apply_reference_to_moving(self, positions: Dict[Tuple[int], gemmi.Position]) -> Dict[
        Tuple[int], gemmi.Position]:
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

@dataclass()
class Alignment:
    transforms: Dict[ResidueID, Transform]

    def __getitem__(self, item: ResidueID):
        return self.transforms[item]

    @staticmethod
    def from_dataset(reference, dataset):

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

                for res in chain.get_polymer():
                    prev_res = chain.previous_residue(res)
                    next_res = chain.next_residue(res)

                    if prev_res:
                        prev_res_id = ResidueID.from_residue_chain(model, chain, prev_res)
                    current_res_id = ResidueID.from_residue_chain(model, chain, res)
                    if next_res:
                        next_res_id = ResidueID.from_residue_chain(model, chain, next_res)

                    if not prev_res:
                        transforms[current_res_id].transform.mat.fromlist(
                            transforms[next_res_id].transform.mat.tolist())

                    if not next_res:
                        transforms[current_res_id].transform.mat.fromlist(
                            transforms[prev_res_id].transform.mat.tolist())

        return Alignment(transforms)

    def __iter__(self):
        for res_id in self.transforms:
            yield res_id
            
