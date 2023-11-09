import time

import numpy as np
import gemmi
from scipy import optimize
from scipy.spatial.transform import Rotation as R
from numpy.random import default_rng

from ..interfaces import *

import os
import inspect

import torch

from .event import Event

from pandda_gemmi.cnn import resnet
from pandda_gemmi.cnn import resnet18
from pandda_gemmi.dmaps import SparseDMap

def sample_xmap(xmap, transform, sample_array):
    xmap.interpolate_values(sample_array, transform)
    return sample_array


def get_sample_transform_from_event(centroid,
                                    sample_distance: float,
                                    n: int,):
    # Get basic sample grid transform
    initial_transform = gemmi.Transform()
    scale_matrix = np.eye(3) * sample_distance
    initial_transform.mat.fromlist(scale_matrix.tolist())

    # Get sample grid centroid
    sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2

    # Get centre grid transform
    centre_grid_transform = gemmi.Transform()
    centre_grid_transform.vec.fromlist([
        -sample_grid_centroid[0],
        -sample_grid_centroid[1],
        -sample_grid_centroid[2],
    ])

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist([x for x in centroid])

    # Get the transform as a gemmi object
    transform = gemmi.Transform()
    transform.vec.fromlist([
        centroid[j] - sample_grid_centroid[j]
        for j
        in [0, 1, 2]
    ])
    transform.mat.fromlist(scale_matrix.tolist())

    corner_0_pos = transform.apply(gemmi.Position(0.0, 0.0, 0.0))
    corner_n_pos = transform.apply(gemmi.Position(
        float(n),
        float(n),
        float(n),
    )
    )
    corner_0 = (corner_0_pos.x, corner_0_pos.y, corner_0_pos.z)
    corner_n = (corner_n_pos.x, corner_n_pos.y, corner_n_pos.z)
    average_pos = [c0 + (cn - c0) / 2 for c0, cn in zip(corner_0, corner_n)]
    event_centroid = [x for x in centroid]
    # print(f"Centroid: {event_centroid}")
    # print(f"Corners: {corner_0} : {corner_n} : average: {average_pos}")

    return transform

# def get_sample_transform_from_event(centroid,
#                                     sample_distance: float,
#                                     n: int,):
#     # Get basic sample grid transform
#     initial_transform = gemmi.Transform()
#     scale_matrix = np.eye(3) * sample_distance
#     initial_transform.mat.fromlist(scale_matrix.tolist())
#
#     # Get sample grid centroid
#     sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2
#
#     # Get centre grid transform
#     centre_grid_transform = gemmi.Transform()
#     centre_grid_transform.vec.fromlist([
#         -sample_grid_centroid[0],
#         -sample_grid_centroid[1],
#         -sample_grid_centroid[2],
#     ])
#
#     # Event centre transform
#     event_centre_transform = gemmi.Transform()
#     event_centre_transform.vec.fromlist([x for x in centroid])
#
#     # Apply random translation
#     transform = event_centre_transform.combine(
#         centre_grid_transform.combine(
#             initial_transform
#         )
#     )
#
#     corner_0_pos = transform.apply(gemmi.Position(0.0, 0.0, 0.0))
#     corner_n_pos = transform.apply(gemmi.Position(
#         float(n),
#         float(n),
#         float(n),
#     )
#     )
#     corner_0 = (corner_0_pos.x, corner_0_pos.y, corner_0_pos.z)
#     corner_n = (corner_n_pos.x, corner_n_pos.y, corner_n_pos.z)
#     average_pos = [c0 + (cn - c0) / 2 for c0, cn in zip(corner_0, corner_n)]
#     event_centroid = [x for x in centroid]
#     print(f"Centroid: {event_centroid}")
#     print(f"Corners: {corner_0} : {corner_n} : average: {average_pos}")
#
#
#     return transform, np.zeros((n, n, n), dtype=np.float32)


def get_model_map(structure, xmap_event):
    # structure = reference.dataset.structure.structure
    new_xmap = gemmi.FloatGrid(xmap_event.nu, xmap_event.nv, xmap_event.nw)
    new_xmap.spacegroup = xmap_event.spacegroup
    new_xmap.set_unit_cell(xmap_event.unit_cell)
    for model in structure:
        for chain in model:
            for residue in chain.get_polymer():
                for atom in residue:
                    new_xmap.set_points_around(
                        atom.pos,
                        radius=1,
                        value=1.0,
                    )

    return new_xmap


def get_correlation(arr1, arr2):
    corr = np.corrcoef(
        np.concatenate(
            (
                arr1.reshape(-1, 1),
                arr2.reshape(-1, 1)
            ),
            axis=1,
        )
    )[0, 1]
    return corr

def get_contrast(bdc, xmap_event_vals, mean_map_event_vals, xmap_inner_vals, mean_map_inner_vals ):
    event_map_event_vals = (xmap_event_vals-(bdc*mean_map_event_vals)) / (1-bdc)
    event_map_inner_vals = (xmap_inner_vals-(bdc*mean_map_inner_vals)) / (1-bdc)

    contrast = get_correlation(event_map_inner_vals, mean_map_inner_vals) - get_correlation(event_map_event_vals, mean_map_event_vals)

    return contrast


def get_bdc(event, xmap_grid, mean_grid, median, reference_frame: DFrameInterface):
    # Get arrays of the xmap and mean map
    xmap_array = np.array(xmap_grid, copy=False)
    mean_array = np.array(mean_grid, copy=False)

    # Get the indicies corresponding to event density i.e. a selector of event density
    event_indicies = tuple(
        [
            event.point_array[:, 0].flatten(),
            event.point_array[:, 1].flatten(),
            event.point_array[:, 2].flatten(),
        ]
    )

    # Get the values of the 2Fo-Fc map and mean map for the event
    xmap_vals = xmap_array[event_indicies]
    mean_map_vals = mean_array[event_indicies]

    # Get the inner mask indicies
    # xmap_inner_vals = reference_frame.mask_inner(xmap_grid).vals
    # mean_inner_vals = reference_frame.mask_inner(mean_grid).vals

    # Get the BDC by minimizing the difference between masked event map density and the median of protein density
    res = optimize.minimize(
        lambda _bdc: np.abs(
            np.median(
                (xmap_vals - (_bdc * mean_map_vals)) / (1 - _bdc)
            ) - median
        ),
        # lambda _bdc: -get_contrast(
        #     _bdc,
        #     xmap_vals,
        #     mean_map_vals,
        #     xmap_inner_vals,
        #     mean_inner_vals
        # ),
        0.5,
        bounds=((0.0, 0.95),),
        tol=0.1
    )

    return float(res.x)



class ScoreCNN:
    def __init__(self, n=30):
        # Get model
        if torch.cuda.is_available():
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

        # Load the model
        cnn = resnet18(num_classes=2, num_input=4)
        cnn_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model.pt"
        cnn.load_state_dict(torch.load(cnn_path, map_location=self.dev))

        # Add model to device
        cnn.to(self.dev)
        cnn.eval()
        self.cnn = cnn.float()

        self.n = n


    def __call__(self, events, xmap_grid, mean_grid, z_grid, model_grid, median):

        scored_events = {}
        time_begin_get_images = time.time()
        images = {}
        bdcs = {}
        for event_id, event in events.items():
            centroid = np.mean(event.pos_array, axis=0)
            dist = np.linalg.norm(centroid - [6.0, -4.0, 25.0])
            # if dist < 5.0:
            #     print(f"##### {event_id} #####")
            #     print(f"Centroid: {centroid}")
            #     print(f"Distance: {dist}")
            sample_transform = get_sample_transform_from_event(
                centroid,
                0.5,
                self.n,
            )

            # Get the BDC
            bdc = get_bdc(event, xmap_grid, mean_grid, median)
            bdcs[event_id] = bdc

            sample_array = np.zeros((self.n, self.n, self.n), dtype=np.float32)


            xmap_sample = sample_xmap(xmap_grid, sample_transform, np.copy(sample_array))

            mean_map_sample = sample_xmap(mean_grid, sample_transform, np.copy(sample_array))

            sample_event = (xmap_sample - (bdc * mean_map_sample)) / (1 - bdc)
            image_event = sample_event[np.newaxis, :]
            image_raw = xmap_sample[np.newaxis, :]


            sample_array_zmap = np.copy(sample_array)
            zmap_sample = sample_xmap(z_grid, sample_transform, sample_array_zmap)
            image_zmap = (zmap_sample[np.newaxis, :] - np.mean())

            sample_array_model = np.copy(sample_array)

            model_sample = sample_xmap(model_grid, sample_transform, sample_array_model)
            image_model = model_sample[np.newaxis, :]

            image = np.stack([image_event, image_raw, image_zmap, image_model], axis=1)
            images[event_id] = image


        for event_id, event in events.items():
            image = images[event_id]

            # Transfer to tensor
            image_t = torch.from_numpy(image)

            # Move tensors to device
            image_c = image_t.to(self.dev)

            # Run model
            model_annotation = self.cnn(image_c.float())

            # Track score
            model_annotations = model_annotation.to(torch.device("cpu")).detach().numpy()

            # flat_bdcs = bdcs.flatten()
            max_score_index = np.argmax([annotation for annotation in model_annotations[:, 1]])
            score = float(model_annotations[max_score_index, 1])

            scored_event = Event(
                event.pos_array,
                event.point_array,
                event.centroid,
                score,
                round(float(bdcs[event_id]), 2)
            )
            scored_events[event_id] = scored_event

        return scored_events



def parse_pdb_file_for_ligand_array(path):
    structure = gemmi.read_structure(str(path))
    poss = []
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    pos = atom.pos
                    poss.append([pos.x, pos.y, pos.z])

    return np.array(poss).T


def get_ligand_map_from_path(path, n, step, translation):
    # Get the ligand array
    ligand_array = parse_pdb_file_for_ligand_array(path)

    if ligand_array.size < 6:
        return None

    rotation_matrix = R.random().as_matrix()
    rng = default_rng()
    random_translation = ((rng.random(3) - 0.5) * 2 * translation).reshape((3, 1))
    ligand_mean_pos = np.mean(ligand_array, axis=1).reshape((3, 1))
    centre_translation = np.array([step * n, step * n, step * n]).reshape((3, 1)) / 2
    zero_centred_array = ligand_array - ligand_mean_pos
    rotated_array = np.matmul(rotation_matrix, zero_centred_array)
    grid_centred_array = rotated_array + centre_translation
    augmented_array = (grid_centred_array + random_translation).T

    # Get a dummy grid to place density on
    dummy_grid = gemmi.FloatGrid(n, n, n)
    unit_cell = gemmi.UnitCell(step * n, step * n, step * n, 90.0, 90.0, 90.0)
    dummy_grid.set_unit_cell(unit_cell)

    for pos_array in augmented_array:
        # assert pos_array.size == 3
        if np.all(pos_array > 0):
            if np.all(pos_array < (n * step)):
                dummy_grid.set_points_around(
                    gemmi.Position(*pos_array),
                    radius=1.0,
                    value=1.0,
                )

    return dummy_grid


def get_ligand_map_from_dataset(
        event,
        n=30,
        step=0.5,
        translation=2.5,
):
    # Get the path to the ligand cif
    dataset_dir = Path(event.model_building_dir) / event.dtag / "compound"
    pdb_paths = [x for x in dataset_dir.glob("*.pdb") if x.exists()]
    path = pdb_paths[0]

    ligand_map = get_ligand_map_from_path(path, n, step, translation)

    return ligand_map

def get_ligand_map_from_ligand_files(
        all_ligand_files,
        n=30,
        step=0.5,
        translation=2.5,
):
    # Get the paths to the first ligand cifs with a pdb to load and use that pdb for the structure
    for key, ligand_files in all_ligand_files.items():
        if ligand_files.ligand_cif:
            if ligand_files.ligand_pdb:

                path = ligand_files.ligand_pdb
                # print(f"Getting ligand map from file: {path}")
                ligand_map = get_ligand_map_from_path(path, n, step, translation)
                if ligand_map is not None:
                    return ligand_map

    # print(f"Couldn't find ligand files!")
    return None


class ScoreCNNLigand:
    def __init__(self, n=30):
        # Get model
        if torch.cuda.is_available():
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

        # Load the model
        # cnn = resnet18(num_classes=2, num_input=4)
        cnn = resnet18(num_classes=2, num_input=4)

        cnn_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model_ligand.pt"
        cnn.load_state_dict(torch.load(cnn_path, map_location=self.dev))

        # Add model to device
        cnn.to(self.dev)
        cnn.eval()
        self.cnn = cnn#.float()

        self.n = n

    def __call__(
            self,
            ligand_files,
            events, homogenized_xmap_grid, xmap_grid, mean_grid, z_grid, model_grid, median, reference_frame,
                 dtag_array, mean
                               ):

        scored_events = {}
        time_begin_get_images = time.time()
        images = {}
        bdcs = {}

        # Get a template for the array to sample onto
        sample_array = np.zeros((self.n, self.n, self.n), dtype=np.float32)

        # Get an image representation of the ligand structure
        ligand_map = get_ligand_map_from_ligand_files(ligand_files)
        if ligand_map is not None:
            image_ligand = np.array(ligand_map)[np.newaxis, :]
        else:
            image_ligand = np.copy(sample_array)[np.newaxis, :]

        # Get the image to Score for each event
        for event_id, event in events.items():
            # print(event_id)
            # Get the estimated bdc for the event (the fraction of the mean map to subtract)
            bdc = get_bdc(event, homogenized_xmap_grid, mean_grid, median, reference_frame)
            bdcs[event_id] = bdc

            # Get the transform that will centre a sampling window on the event
            centroid = np.mean(event.pos_array, axis=0)
            sample_transform = get_sample_transform_from_event(
                centroid,
                0.5,
                self.n,
            )

            # Get images of the 2Fo-Fc map, the ground state (mean) map and the structure map
            sample_array_raw = np.copy(sample_array)
            xmap_sample = sample_xmap(xmap_grid, sample_transform, sample_array_raw)
            xmap_mean = np.mean(xmap_sample)
            xmap_std = np.std(xmap_sample)
            image_xmap = (xmap_sample[np.newaxis, :] - xmap_mean) / xmap_std
            # print(f"Xmap: {[xmap_mean, xmap_std]}")
            #
            # sample_array_mean_map = np.copy(sample_array)
            # mean_sample = sample_xmap(mean_grid, sample_transform, sample_array_mean_map)
            # mean_mean = np.mean(mean_sample)
            # mean_std = np.std(mean_sample)
            # image_mean = (mean_sample[np.newaxis, :] - mean_mean) / mean_std
            # print(f"Mean: {[mean_mean, mean_std]}")

            event_array = (dtag_array - (event.bdc * mean)) / (1 - event.bdc)
            event_map_grid = reference_frame.unmask(SparseDMap(event_array))

            # mean_grid_array = np.array(mean_grid, copy=False)
            # homogenized_xmap_grid_array = np.array(homogenized_xmap_grid, copy=False)
            # event_map_array = (homogenized_xmap_grid_array - (event.bdc * mean_grid_array)) / (1 - event.bdc)
            #
            # event_map_grid = reference_frame.get_grid()
            # event_map_grid_array = np.array(event_map_grid, copy=False)
            # event_map_grid_array[:,:,:] = event_map_array[:,:,:]
            # event_map_grid = reference_frame.unmask(SparseDMap(event_map_array))
            sample_array_event_map = np.copy(sample_array)
            event_map_sample = sample_xmap(event_map_grid, sample_transform, sample_array_event_map)
            event_map_mean = np.mean(event_map_sample)
            event_map_std = np.std(event_map_sample)
            image_event_map = (event_map_sample[np.newaxis, :] - event_map_mean) / event_map_std

            sample_array_model = np.copy(sample_array)
            model_sample = sample_xmap(model_grid, sample_transform, sample_array_model)
            image_model = model_sample[np.newaxis, :]
            # print(f"Model: {np.mean(image_model)}")
            # print(f"Ligand: {np.mean(image_ligand)}")

            # Generate the combined image for scoring
            image = np.stack([image_xmap, image_event_map, image_model, image_ligand, ], axis=1)
            # image = np.stack([image_xmap, image_mean, image_model, image_ligand, ], axis=1)
            # image = np.stack([image_event_map, image_model, image_ligand, ], axis=1)
            images[event_id] = image

        time_finish_get_images = time.time()
        # print(f"\t\t\t\tGot images in: {round(time_finish_get_images - time_begin_get_images, 2)}")


        # Score each event
        cnn_times = []
        time_begin_score_images = time.time()
        for event_id, event in events.items():
            image = images[event_id]

            # Transfer to tensor
            image_t = torch.from_numpy(image)

            # Move tensors to device
            image_c = image_t.to(self.dev)#.float()

            # Run model
            time_begin_cnn = time.time()
            model_annotation = self.cnn(image_c)
            time_finish_cnn = time.time()
            cnn_times.append(time_finish_cnn-time_begin_cnn)

            # Track score
            model_annotations = model_annotation.to(torch.device("cpu")).detach().numpy()
            max_score_index = np.argmax([annotation for annotation in model_annotations[:, 1]])
            score = float(model_annotations[max_score_index, 1])

            # Generate an event with the score and BDC
            scored_event = Event(
                event.pos_array,
                event.point_array,
                event.centroid,
                score,
                round(float(bdcs[event_id]), 2)
            )
            scored_events[event_id] = scored_event
        time_finish_score_images = time.time()
        # print(f"\t\t\t\tScored images in: {round(time_finish_score_images - time_begin_score_images, 2)} of which {round(sum(cnn_times), 2)} was in cnn")

        return scored_events
