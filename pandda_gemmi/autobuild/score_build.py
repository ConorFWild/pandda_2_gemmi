import time

import numpy as np
import gemmi

from ..interfaces import *

import os
import inspect

import torch
from torch.nn import functional as F
#import lightning as lt
import pytorch_lightning as lt
import yaml


from pandda_gemmi.cnn import resnet
from pandda_gemmi.cnn import resnet18, resnet10


def _get_identity_matrix():
    return np.eye(3)


def _get_centroid_from_res(res):
    poss = []
    for atom in res:
        pos = atom.pos
        poss.append([pos.x, pos.y, pos.z])

    return np.mean(poss, axis=0)


def combine(new_transform, old_transform):
    new_transform_mat = new_transform.mat
    new_transform_vec = new_transform.vec

    old_transform_mat = old_transform.mat
    old_transform_vec = old_transform.vec

    combined_transform_mat = new_transform_mat.multiply(old_transform_mat)
    combined_transform_vec = new_transform_vec + new_transform_mat.multiply(old_transform_vec)

    combined_transform = gemmi.Transform()
    combined_transform.vec.fromlist(combined_transform_vec.tolist())
    combined_transform.mat.fromlist(combined_transform_mat.tolist())

    return combined_transform


# def _get_transform_from_orientation_centroid(orientation, centroid):
#     sample_distance: float = 0.5
#     n: int = 30
#     # translation: float):
#
#     # Get basic sample grid transform
#     initial_transform = gemmi.Transform()
#     scale_matrix = np.eye(3) * sample_distance
#     initial_transform.mat.fromlist(scale_matrix.tolist())
#
#     # Get sample grid centroid
#     sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2
#     sample_grid_centroid_pos = gemmi.Position(*sample_grid_centroid)
#
#     # Get centre grid transform
#     centre_grid_transform = gemmi.Transform()
#     centre_grid_transform.vec.fromlist([
#         -sample_grid_centroid[0],
#         -sample_grid_centroid[1],
#         -sample_grid_centroid[2],
#     ])
#
#     # Generate rotation matrix
#     rotation_matrix = orientation
#     rotation_transform = gemmi.Transform()
#     rotation_transform.mat.fromlist(rotation_matrix.tolist())
#
#     # Apply random rotation transform to centroid
#     transformed_centroid = rotation_transform.apply(sample_grid_centroid_pos)
#     transformed_centroid_array = np.array([transformed_centroid.x, transformed_centroid.y, transformed_centroid.z])
#
#     # Recentre transform
#     rotation_recentre_transform = gemmi.Transform()
#     rotation_recentre_transform.vec.fromlist((sample_grid_centroid - transformed_centroid_array).tolist())
#
#     # Event centre transform
#     event_centre_transform = gemmi.Transform()
#     event_centre_transform.vec.fromlist(centroid)
#
#     # Apply random translation
#     # transform = event_centre_transform.combine(
#     #     rotation_transform.combine(
#     #         centre_grid_transform.combine(
#     #             initial_transform
#     #         )
#     #     )
#     # )
#
#     transform = combine(
#         event_centre_transform,
#         combine(
#             rotation_transform,
#             combine(
#                 centre_grid_transform,
#                 initial_transform)))
#     return transform

def _get_transform_from_orientation_centroid(orientation, centroid, n, sd=0.5):
    sample_distance: float = sd

    # Get basic sample grid transform
    initial_transform = gemmi.Transform()
    scale_matrix = np.eye(3) * sample_distance
    initial_transform.mat.fromlist(scale_matrix.tolist())

    # Get sample grid centroid
    sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2
    # sample_grid_centroid_pos = gemmi.Position(*sample_grid_centroid)

    # Get centre grid transform
    centre_grid_transform = gemmi.Transform()
    centre_grid_transform.vec.fromlist([
        -sample_grid_centroid[0],
        -sample_grid_centroid[1],
        -sample_grid_centroid[2],
    ])

    # Generate rotation matrix
    rotation_matrix = orientation
    rotation_transform = gemmi.Transform()
    rotation_transform.mat.fromlist(rotation_matrix.tolist())

    # Apply random rotation transform to centroid
    # transformed_centroid = rotation_transform.apply(sample_grid_centroid_pos)
    # transformed_centroid_array = np.array([transformed_centroid.x, transformed_centroid.y, transformed_centroid.z])

    # Recentre transform
    # rotation_recentre_transform = gemmi.Transform()
    # rotation_recentre_transform.vec.fromlist((sample_grid_centroid - transformed_centroid_array).tolist())

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist(centroid)

    transform = combine(
        event_centre_transform,
        combine(
            rotation_transform,
            combine(
                centre_grid_transform,
                initial_transform)))
    return transform


def get_ligand_mask(dmap, res):
    mask = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P1")
    mask.set_unit_cell(dmap.unit_cell)

    # Get the mask
    for atom in res:
        pos = atom.pos
        mask.set_points_around(
            pos,
            radius=2.5,
            value=1,
        )

    return mask


def get_ligand_mask_float(res, radius, n, r):

    mask = gemmi.FloatGrid(n, n, n)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P1")
    mask.set_unit_cell(gemmi.UnitCell(r, r, r, 90.0, 90.0, 90.0))

    # Get the mask
    for atom in res:
        pos = atom.pos
        mask.set_points_around(
            pos,
            radius=radius,
            value=1.0,
        )

    return mask


def get_masked_dmap(dmap, res):
    mask = get_ligand_mask(dmap, res)

    # Get the mask array
    mask_array = np.array(mask, copy=False)

    # Get the dmap array
    dmap_array = np.array(dmap, copy=False)

    # Mask the dmap array
    dmap_array[mask_array == 0] = 0.0

    return dmap


def sample_xmap(xmap, transform, sample_array):
    xmap.interpolate_values(sample_array, transform)
    return sample_array


def sample_xmap_and_scale(masked_dmap, sample_transform, sample_array):
    image_initial = sample_xmap(masked_dmap, sample_transform, sample_array)
    std = np.std(image_initial)
    if np.abs(std) < 0.0000001:
        image_dmap = np.copy(sample_array)

    else:
        image_dmap = (image_initial - np.mean(image_initial)) / std

    return image_dmap


def _make_ligand_masked_dmap_layer(
        dmap,
        res,
        sample_transform,
        sample_array
):
    # Get the masked dmap

    masked_dmap = get_masked_dmap(dmap, res)

    # Get the image
    image_dmap = sample_xmap_and_scale(masked_dmap, sample_transform, sample_array)

    return image_dmap


# class ScoreCNNEventBuild:
#     def __init__(self, n=30):
#         # Get model
#         if torch.cuda.is_available():
#             self.dev = "cuda:0"
#         else:
#             self.dev = "cpu"
#
#         # Load the model
#         cnn = resnet18(num_classes=2, num_input=3)
#
#         cnn_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model_event_build.pt"
#         cnn.load_state_dict(torch.load(cnn_path, map_location=self.dev))
#
#         # Add model to device
#         cnn.to(self.dev)
#         cnn.eval()
#         self.cnn = cnn  # .float()
#
#         self.n = n
#
#     def __call__(
#             self,
#             optimized_structure,
#             corrected_event_map_grid,
#             z_grid,
#             raw_xmap_grid,
#     ):
#         res = optimized_structure[0][0][0]
#         rotation = _get_identity_matrix()
#         centroid = _get_centroid_from_res(res)
#         transform = _get_transform_from_orientation_centroid(rotation, centroid)
#         sample_array = np.zeros(
#             (30, 30, 30),
#             dtype=np.float32,
#         )
#
#         image_event_map = _make_ligand_masked_dmap_layer(
#             corrected_event_map_grid,
#             res,
#             transform,
#             sample_array
#         )
#         image_z_map = _make_ligand_masked_dmap_layer(
#             z_grid,
#             res,
#             transform,
#             sample_array
#         )
#         image_raw_xmap = _make_ligand_masked_dmap_layer(
#             raw_xmap_grid,
#             res,
#             transform,
#             sample_array
#         )
#         image = torch.tensor(
#             np.stack(
#                 [image_event_map, image_z_map, image_raw_xmap],
#                 axis=0
#             )[np.newaxis, :]
#         )
#         # rprint(f"Image shape is: {image.shape}")
#
#         # Get the device
#         if torch.cuda.is_available():
#             dev = "cuda:0"
#         else:
#             dev = "cpu"
#
#         image_c = image.to(dev)
#         annotation = self.cnn(image_c)
#
#         return float(annotation[0][1])


class LitBuildScoring(lt.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet = resnet10(num_classes=1, num_input=2).float()
        # self.train_annotations = []
        # self.test_annotations = []
        # self.output = Path('./output/build_scoring_hdf5')

    def forward(self, x):
        return F.sigmoid(self.resnet(x))


# class ScoreCNNEventBuildAllLayers:
#     def __init__(self, n=30):
#         # Get model
#         if torch.cuda.is_available():
#             self.dev = "cuda:0"
#         else:
#             self.dev = "cpu"
#
#         # Load the model
#         cnn = resnet18(num_classes=2, num_input=3)
#
#         cnn_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model_event_build.pt"
#         cnn.load_state_dict(torch.load(cnn_path, map_location=self.dev))
#
#         # Add model to device
#         cnn.to(self.dev)
#         cnn.eval()
#         self.cnn = cnn  # .float()
#
#         self.n = n
#
#     def __call__(
#             self,
#             optimized_structure,
#             xmap,
#             mean_map,
#             bdc,
#             z_grid,
#             raw_xmap_grid,
#     ):
#         res = optimized_structure[0][0][0]
#         rotation = _get_identity_matrix()
#         centroid = _get_centroid_from_res(res)
#         transform = _get_transform_from_orientation_centroid(rotation, centroid)
#         sample_array = np.zeros(
#             (30, 30, 30),
#             dtype=np.float32,
#         )
#
#         image_dmap = sample_xmap_and_scale(
#             xmap,
#             # res,
#             transform,
#             np.copy(sample_array)
#         )
#         image_mean_map = sample_xmap_and_scale(
#             mean_map,
#             # res,
#             transform,
#             np.copy(sample_array)
#         )
#         image_z_map = sample_xmap_and_scale(
#             z_grid,
#             # res,
#             transform,
#             np.copy(sample_array)
#         )
#         image_raw_xmap = sample_xmap_and_scale(
#             raw_xmap_grid,
#             # res,
#             transform,
#             np.copy(sample_array)
#         )
#         ligand_mask_grid = get_ligand_mask_float(xmap, res)
#         image_ligand_mask = sample_xmap(
#             ligand_mask_grid,
#             # res,
#             transform,
#             np.copy(sample_array)
#         )
#         image_ligand_mask[image_ligand_mask < 0.9] = 0.0
#         image_ligand_mask[image_ligand_mask > 0.9] = 1.0
#
#         image = np.stack(
#             [
#                 ((image_dmap - (bdc * image_mean_map)) / (1 - bdc)) * image_ligand_mask,
#                 image_z_map * image_ligand_mask,
#                 image_raw_xmap * image_ligand_mask,
#             ],
#             axis=0,
#         )[np.newaxis, :]
#         image_float = image.astype(np.float32)
#
#         # print(f"image dtype: {image_float.dtype}, Image shape: {image_float.shape}")
#
#         image_t = torch.from_numpy(image_float)
#         image_c = image_t.to(self.dev)
#         # print(f"Image tensor dtype: {image_c.dtype}, image tensor shape: {image_c.shape}")
#
#         cnn = self.cnn.float()
#
#         model_annotation = cnn(image_c)
#         # print(f'Annotation shape: {model_annotation.shape}')
#
#         annotation = model_annotation.to(torch.device("cpu")).detach().numpy()
#
#         return float(annotation[0][1])


class ScoreCNNEventBuild:
    def __init__(self, n=30):
        # Get model
        if torch.cuda.is_available():
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

        config_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model_event_build_config.yaml"
        self.config = yaml.safe_load(config_path)

        # Load the model
        cnn = LitBuildScoring()

        cnn_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model_event_build.pt"
        cnn.load_state_dict(torch.load(cnn_path, map_location=self.dev)['state_dict'])

        # Add model to device
        cnn.to(self.dev)
        cnn.eval()
        self.cnn = cnn  # .float()

        self.n = n



    def __call__(
            self,
            optimized_structure,
            xmap,
            mean_map,
            bdc,
            z_grid,
            raw_xmap_grid,
    ):
        res = optimized_structure[0][0][0]
        rotation = _get_identity_matrix()
        centroid = _get_centroid_from_res(res)
        transform = _get_transform_from_orientation_centroid(rotation, centroid, n=self.config['grid_size'], sd=self.config['grid_step'])
        sample_array = np.zeros(
            (self.config['grid_size'], self.config['grid_size'], self.config['grid_size']),
            dtype=np.float32,
        )

        # image_dmap = sample_xmap(
        #     xmap,
        #     transform,
        #     np.copy(sample_array)
        # )
        # image_mean_map = sample_xmap(
        #     mean_map,
        #     transform,
        #     np.copy(sample_array)
        # )
        image_z_map = sample_xmap(
            z_grid,
            transform,
            np.copy(sample_array)
        )
        image_raw_xmap = sample_xmap(
            raw_xmap_grid,
            transform,
            np.copy(sample_array)
        )

        ligand_mask_grid = get_ligand_mask_float(xmap, res)
        image_ligand_mask = sample_xmap(
            ligand_mask_grid,
            transform,
            np.copy(sample_array)
        )
        image_ligand_mask[image_ligand_mask < 0.5] = 0.0
        image_ligand_mask[image_ligand_mask >= 0.5] = 1.0

        # image = np.stack(
        #     [
        #         ((image_dmap - (bdc * image_mean_map)) / (1 - bdc)) * image_ligand_mask,
        #         image_raw_xmap * image_ligand_mask
        #     ],
        #     axis=0,
        # )[np.newaxis, :]
        image = np.stack(
            [
                image_z_map * image_ligand_mask,
                image_raw_xmap * image_ligand_mask,
                image_ligand_mask
            ],
            axis=0,
        )[np.newaxis, :]
        image_float = image.astype(np.float32)

        image_t = torch.from_numpy(image_float)
        image_c = image_t.to(self.dev)

        cnn = self.cnn.float()

        model_annotation = cnn(image_c)

        annotation = model_annotation.to(torch.device("cpu")).detach().numpy()

        return annotation[0]
