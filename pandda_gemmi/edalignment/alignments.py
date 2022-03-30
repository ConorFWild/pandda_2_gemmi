import typing
import dataclasses

import scipy
from scipy import spatial
from joblib.externals.loky import set_loky_pickler
from pandda_gemmi.analyse_interface import AlignmentsInterface, DatasetsInterface, GetAlignmentsInterface, ReferenceInterface

set_loky_pickler('pickle')

from pandda_gemmi.python_types import *
from pandda_gemmi.pandda_exceptions import *
from pandda_gemmi.common import Dtag
from pandda_gemmi.dataset import Dataset, ResidueID, Reference, Datasets


@dataclasses.dataclass()
class Transform:
    transform: gemmi.Transform
    com_reference: np.array
    com_moving: np.array

    def apply_moving_to_reference(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
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

    def apply_reference_to_moving(self, positions: typing.Dict[typing.Tuple[int], gemmi.Position]) -> typing.Dict[
        typing.Tuple[int], gemmi.Position]:
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

    # @staticmethod
    # def from_residues(previous_res, current_res, next_res, previous_ref, current_ref, next_ref):
    #     previous_ca_pos = previous_res["CA"][0].pos
    #     current_ca_pos = current_res["CA"][0].pos
    #     next_ca_pos = next_res["CA"][0].pos

    #     previous_ref_ca_pos = previous_ref["CA"][0].pos
    #     current_ref_ca_pos = current_ref["CA"][0].pos
    #     next_ref_ca_pos = next_ref["CA"][0].pos

    #     matrix = np.array([
    #         Transform.pos_to_list(previous_ca_pos),
    #         Transform.pos_to_list(current_ca_pos),
    #         Transform.pos_to_list(next_ca_pos),
    #     ])
    #     matrix_ref = np.array([
    #         Transform.pos_to_list(previous_ref_ca_pos),
    #         Transform.pos_to_list(current_ref_ca_pos),
    #         Transform.pos_to_list(next_ref_ca_pos),
    #     ])

    #     mean = np.mean(matrix, axis=0)
    #     mean_ref = np.mean(matrix_ref, axis=0)

    #     # vec = mean_ref - mean
    #     vec = np.array([0.0, 0.0, 0.0])

    #     de_meaned = matrix - mean
    #     de_meaned_ref = matrix_ref - mean_ref

    #     rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

    #     com_reference = mean_ref
    #     com_moving = mean

    #     return Transform.from_translation_rotation(vec, rotation, com_reference, com_moving)

    @staticmethod
    def pos_to_list(pos: gemmi.Position):
        return [pos[0], pos[1], pos[2]]

    # @staticmethod
    # def from_start_residues(current_res, next_res, current_ref, next_ref):
    #     current_ca_pos = current_res["CA"][0].pos
    #     next_ca_pos = next_res["CA"][0].pos

    #     current_ref_ca_pos = current_ref["CA"][0].pos
    #     next_ref_ca_pos = next_ref["CA"][0].pos

    #     matrix = np.array([
    #         Transform.pos_to_list(current_ca_pos),
    #         Transform.pos_to_list(next_ca_pos),
    #     ])
    #     matrix_ref = np.array([
    #         Transform.pos_to_list(current_ref_ca_pos),
    #         Transform.pos_to_list(next_ref_ca_pos),
    #     ])

    #     mean = np.mean(matrix, axis=0)
    #     mean_ref = np.mean(matrix_ref, axis=0)

    #     # vec = mean_ref - mean
    #     vec = np.array([0.0, 0.0, 0.0])

    #     de_meaned = matrix - mean
    #     de_meaned_ref = matrix_ref - mean_ref

    #     rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

    #     com_reference = mean_ref

    #     com_moving = mean

    #     return Transform.from_translation_rotation(vec, rotation, com_reference, com_moving)

    @staticmethod
    def from_atoms(dataset_selection,
                   reference_selection,
                   com_dataset,
                   com_reference,
                   ):

        # mean = np.mean(dataset_selection, axis=0)
        # mean_ref = np.mean(reference_selection, axis=0)
        # mean = np.array(com_dataset)
        # mean_ref = np.array(com_reference)
        mean = com_dataset
        mean_ref = com_reference

        # vec = mean_ref - mean
        vec = np.array([0.0, 0.0, 0.0])

        de_meaned = dataset_selection - mean
        de_meaned_ref = reference_selection - mean_ref

        rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

        com_reference = mean_ref

        com_moving = mean

        return Transform.from_translation_rotation(vec, rotation, com_reference, com_moving)

    # @staticmethod
    # def from_finish_residues(previous_res, current_res, previous_ref, current_ref):
    #     previous_ca_pos = previous_res["CA"][0].pos
    #     current_ca_pos = current_res["CA"][0].pos

    #     previous_ref_ca_pos = previous_ref["CA"][0].pos
    #     current_ref_ca_pos = current_ref["CA"][0].pos

    #     matrix = np.array([
    #         Transform.pos_to_list(previous_ca_pos),
    #         Transform.pos_to_list(current_ca_pos),
    #     ])
    #     matrix_ref = np.array([
    #         Transform.pos_to_list(previous_ref_ca_pos),
    #         Transform.pos_to_list(current_ref_ca_pos),
    #     ])

    #     mean = np.mean(matrix, axis=0)
    #     mean_ref = np.mean(matrix_ref, axis=0)

    #     # vec = mean_ref - mean
    #     vec = np.array([0.0, 0.0, 0.0])

    #     de_meaned = matrix - mean
    #     de_meaned_ref = matrix_ref - mean_ref

    #     rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned, de_meaned_ref)

    #     com_reference = mean_ref

    #     com_moving = mean

    #     return Transform.from_translation_rotation(vec, rotation, com_reference, com_moving)

    def __getstate__(self):
        transform_python = TransformPython.from_gemmi(self.transform)
        com_reference = self.com_reference
        com_moving = self.com_moving
        return (transform_python, com_reference, com_moving)

    def __setstate__(self, data):
        transform_gemmi = data[0].to_gemmi()
        self.transform = transform_gemmi
        self.com_reference = data[1]
        self.com_moving = data[2]


@dataclasses.dataclass()
class Alignment:
    transforms: typing.Dict[ResidueID, Transform]

    def __getitem__(self, item: ResidueID):
        return self.transforms[item]

    def reference_to_moving(self, positions):

        transforms_list = [transform for transform in self.transforms.values()]

        reference_positions = np.vstack([transform.com_reference for transform in transforms_list])
        tree = spatial.KDTree(reference_positions)
        dist, inds = tree.query(positions)

        results = []
        for pos, ind in zip(positions, inds):
            transform = transforms_list[ind]
            transformed_pos = transform.apply_reference_to_moving({(0, 0, 0): gemmi.Position(*pos), })[(0, 0, 0)]
            results.append((transformed_pos.x, transformed_pos.y, transformed_pos.z,))

        return results

    @staticmethod
    def has_large_gap(reference: Reference, dataset: Dataset):
        try:
            Alignment.from_dataset(reference, dataset)

        except ExceptionUnmatchedAlignmentMarker as e:
            return False

        except ExceptionNoCommonAtoms as e:
            return False

        return True

    @staticmethod
    def from_dataset(reference: Reference, dataset: Dataset, marker_atom_search_radius=10.0):

        dataset_pos_list = []
        reference_pos_list = []

        # Iterate protein atoms, then pull out their atoms, and search them
        for res_id in reference.dataset.structure.protein_residue_ids():

            # Get the matchable CAs
            try:
                # Get reference residue
                ref_res_span = reference.dataset.structure[res_id]
                ref_res = ref_res_span[0]

                # Get corresponding reses
                dataset_res_span = dataset.structure[res_id]
                dataset_res = dataset_res_span[0]

                # Get the CAs
                atom_ref = ref_res["CA"][0]
                atom_dataset = dataset_res["CA"][0]

                # Get the shared atoms
                reference_pos_list.append([atom_ref.pos.x, atom_ref.pos.y, atom_ref.pos.z, ])
                dataset_pos_list.append([atom_dataset.pos.x, atom_dataset.pos.y, atom_dataset.pos.z, ])

            except Exception as e:
                print(f"WARNING: An exception occured in matching residues for alignment at residue id: {res_id}: {e}")
                continue

        dataset_atom_array = np.array(dataset_pos_list)
        reference_atom_array = np.array(reference_pos_list)

        if (reference_atom_array.shape[0] == 0) or (dataset_atom_array.shape[0] == 0):
            raise ExceptionNoCommonAtoms()

        # dataset kdtree
        dataset_tree = spatial.KDTree(dataset_atom_array)
        # Other kdtree
        reference_tree = spatial.KDTree(reference_atom_array)

        if reference_atom_array.size != dataset_atom_array.size:
            raise AlignmentUnmatchedAtomsError(reference_atom_array,
                                               dataset_atom_array,
                                               )

        transforms = {}

        # Start searching
        for res_id in reference.dataset.structure.protein_residue_ids():
            # Get reference residue
            ref_res_span = reference.dataset.structure[res_id]
            ref_res = ref_res_span[0]

            # Get ca pos in reference model
            reference_ca_pos = ref_res["CA"][0].pos

            # # Get residue span in other dataset
            # dataset_res_span = dataset.structure[res_id]
            # dataset_res = dataset_res_span[0]

            # # Get ca position in moving dataset model
            # dataset_ca_pos = dataset_res["CA"][0].pos

            # dataset selection
            # dataset_indexes = dataset_tree.query_ball_point([dataset_ca_pos.x, dataset_ca_pos.y, dataset_ca_pos.z],
            #                                                 7.0,
            #                      ExceptionNoCommonAtoms                           )
            # dataset_selection = dataset_atom_array[dataset_indexes]

            # other selection
            reference_indexes = reference_tree.query_ball_point(
                [reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z],
                marker_atom_search_radius,
            )
            reference_selection = reference_atom_array[reference_indexes]
            dataset_selection = dataset_atom_array[reference_indexes]

            if dataset_selection.shape[0] == 0:
                raise ExceptionUnmatchedAlignmentMarker(res_id)

            transforms[res_id] = Transform.from_atoms(
                dataset_selection,
                reference_selection,
                # com_dataset=[dataset_ca_pos.x, dataset_ca_pos.y, dataset_ca_pos.z],
                # com_reference=[reference_ca_pos.x, reference_ca_pos.y, reference_ca_pos.z],
                com_dataset=np.mean(dataset_selection, axis=0),
                com_reference=np.mean(reference_selection, axis=0),

            )

        return Alignment(transforms)

    # @staticmethod
    # def from_dataset(reference: Reference, dataset: Dataset):

    #     transforms = {}

    #     for model in dataset.structure.structure:
    #         for chain in model:
    #             for res in chain.get_polymer():
    #                 prev_res = chain.previous_residue(res)
    #                 next_res = chain.next_residue(res)

    #                 if prev_res:
    #                     prev_res_id = ResidueID.from_residue_chain(model, chain, prev_res)
    #                 current_res_id = ResidueID.from_residue_chain(model, chain, res)
    #                 if next_res:
    #                     next_res_id = ResidueID.from_residue_chain(model, chain, next_res)

    #                 if prev_res:
    #                     prev_res_ref = reference.dataset.structure[prev_res_id][0]
    #                 current_res_ref = reference.dataset.structure[current_res_id][0]
    #                 if next_res:
    #                     next_res_ref = reference.dataset.structure[next_res_id][0]

    #                 if not prev_res:
    #                     transform = Transform.from_start_residues(res, next_res,
    #                                                               current_res_ref, next_res_ref)

    #                 if not next_res:
    #                     transform = Transform.from_finish_residues(prev_res, res,
    #                                                                prev_res_ref, current_res_ref)

    #                 if prev_res and next_res:
    #                     transform = Transform.from_residues(prev_res, res, next_res,
    #                                                         prev_res_ref, current_res_ref, next_res_ref,
    #                                                         )

    #                 transforms[current_res_id] = transform

    #             for res in chain.get_polymer():
    #                 prev_res = chain.previous_residue(res)
    #                 next_res = chain.next_residue(res)

    #                 if prev_res:
    #                     prev_res_id = ResidueID.from_residue_chain(model, chain, prev_res)
    #                 current_res_id = ResidueID.from_residue_chain(model, chain, res)
    #                 if next_res:
    #                     next_res_id = ResidueID.from_residue_chain(model, chain, next_res)

    #                 if not prev_res:
    #                     transforms[current_res_id].transform.mat.fromlist(
    #                         transforms[next_res_id].transform.mat.tolist())

    #                 if not next_res:
    #                     transforms[current_res_id].transform.mat.fromlist(
    #                         transforms[prev_res_id].transform.mat.tolist())

    #     return Alignment(transforms)

    def __iter__(self):
        for res_id in self.transforms:
            yield res_id

    # def __getstate__(self):
    #     alignment = AlignmentPython.from_gemmi(self)
    #     return alignment

    # def __setstate__(self, alignment_python: AlignmentPython):
    #     alignment_gemmi = alignment_python.to_gemmi()
    #     self.transforms = alignment_gemmi


@dataclasses.dataclass()
class Alignments:
    alignments: typing.Dict[Dtag, Alignment]

    def __getitem__(self, item):
        return self.alignments[item]

    def __iter__(self):
        for dtag in self.alignments:
            yield dtag
    #
    # def __getstate__(self):
    #
    #     alignments_python = {}
    #     for dtag, alignment in self.alignments.items():
    #         alignment_python = AlignmentPython.from_gemmi(alignment)
    #         alignments_python[dtag] = alignment_python
    #     return alignments_python
    #
    # def __setstate__(self, alignments_python: Dict[Dtag, AlignmentPython]):
    #     self.alignments = {dtag: alignment_python.to_gemmi() for dtag, alignment_python in alignments_python.items()}
    #
    #

def get_alignments(reference: Reference, datasets: Datasets) -> AlignmentsInterface:
    alignments = {}
    for dtag in datasets:
        alignments[dtag] = Alignment.from_dataset(reference, datasets[dtag])

    return Alignments(alignments)



class GetAlignments(GetAlignmentsInterface):
    def __call__(self, 
    reference: ReferenceInterface, 
    datasets: DatasetsInterface) -> AlignmentsInterface:
        return get_alignments(
            reference,
            datasets,
        )