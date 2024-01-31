import time

import numpy as np
import gemmi

from ..interfaces import *

import os
import inspect

import torch


from pandda_gemmi.cnn import resnet
from pandda_gemmi.cnn import resnet18


def _get_identity_matrix():
    return np.eye(3)


def _get_centroid_from_res(res):
    poss = []
    for atom in res:
        pos = atom.pos
        poss.append([pos.x, pos.y, pos.z])

    return np.mean(poss, axis=0)


def _get_transform_from_orientation_centroid(orientation, centroid):
    sample_distance: float = 0.5
    n: int = 30
    # translation: float):

    # Get basic sample grid transform
    initial_transform = gemmi.Transform()
    scale_matrix = np.eye(3) * sample_distance
    initial_transform.mat.fromlist(scale_matrix.tolist())

    # Get sample grid centroid
    sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2
    sample_grid_centroid_pos = gemmi.Position(*sample_grid_centroid)

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
    transformed_centroid = rotation_transform.apply(sample_grid_centroid_pos)
    transformed_centroid_array = np.array([transformed_centroid.x, transformed_centroid.y, transformed_centroid.z])

    # Recentre transform
    rotation_recentre_transform = gemmi.Transform()
    rotation_recentre_transform.vec.fromlist((sample_grid_centroid - transformed_centroid_array).tolist())

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist(centroid)

    # Apply random translation
    transform = event_centre_transform.combine(
        rotation_transform.combine(
            centre_grid_transform.combine(
                initial_transform
            )
        )
    )
    return transform


def get_masked_dmap(dmap, res):
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


def _make_ligand_masked_dmap_layer(
        dmap,
        res,
        sample_transform,
        sample_array
):
    # Get the masked dmap

    masked_dmap = get_masked_dmap(dmap, res)

    # Get the image
    image_initial = sample_xmap(masked_dmap, sample_transform, sample_array)
    std = np.std(image_initial)
    if np.abs(std) < 0.0000001:
        image_dmap = np.copy(sample_array)

    else:
        image_dmap = (image_initial - np.mean(image_initial)) / std

    return image_dmap


class ScoreCNNEventBuild:
    def __init__(self, n=30):
        # Get model
        if torch.cuda.is_available():
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

        # Load the model
        cnn = resnet18(num_classes=2, num_input=3)

        cnn_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model_event_build.pt"
        cnn.load_state_dict(torch.load(cnn_path, map_location=self.dev))

        # Add model to device
        cnn.to(self.dev)
        cnn.eval()
        self.cnn = cnn  # .float()

        self.n = n

    def __call__(
            self,
            optimized_structure,
            corrected_event_map_grid,
            z_grid,
            raw_xmap_grid,
    ):
        res = optimized_structure[0][0][0]
        rotation = _get_identity_matrix()
        centroid = _get_centroid_from_res(res)
        transform = _get_transform_from_orientation_centroid(rotation, centroid)
        sample_array = np.zeros(
            (30, 30, 30),
            dtype=np.float32,
        )

        image_event_map = _make_ligand_masked_dmap_layer(
            corrected_event_map_grid,
            res,
            transform,
            sample_array
        )
        image_z_map = _make_ligand_masked_dmap_layer(
            z_grid,
            res,
            transform,
            sample_array
        )
        image_raw_xmap = _make_ligand_masked_dmap_layer(
            raw_xmap_grid,
            res,
            transform,
            sample_array
        )
        image = torch.tensor(
            np.stack(
                [image_event_map, image_z_map, image_raw_xmap],
                axis=0
            )[np.newaxis, :]
        )
        # rprint(f"Image shape is: {image.shape}")

        # Get the device
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        image_c = image.to(dev)
        annotation = self.cnn(image_c)

        return float(annotation[0][1])
