import shutil

import fire

from typing import Dict, List
import time
from pathlib import Path

import gemmi
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import inspect

import torch

from pandda_gemmi.cnn import resnet
from pandda_gemmi.cnn import resnet18

from pandda_gemmi.scratch.interfaces import *
from pandda_gemmi.scratch.fs import PanDDAFS
from pandda_gemmi.scratch.dataset import XRayDataset, Reflections
from pandda_gemmi.scratch.dmaps import DMap, SparseDMap, SparseDMapStream, TruncateReflections, SmoothReflections
from pandda_gemmi.scratch.alignment import Alignment, DFrame
from pandda_gemmi.scratch.processor import ProcessLocalRay, Partial
from pandda_gemmi.scratch.dataset import ResidueID

def sample_xmap(xmap, transform, sample_array):
    xmap.interpolate_values(sample_array, transform)
    return sample_array


def get_sample_transform_from_event(centroid,
                                    sample_distance: float,
                                    n: int,
                                    translation: float):
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

    # Apply random translation
    # transform = event_centre_transform.combine(
    #     centre_grid_transform.combine(
    #         initial_transform
    #     )
    # )

    transform = gemmi.Transform()
    transform.vec.fromlist([
        centroid[j] - sample_grid_centroid[j]
        for j
        in [0,1,2]
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
    event_centroid = centroid
    # logger.debug(f"Centroid: {event_centroid}")
    # logger.debug(f"Corners: {corner_0} : {corner_n} : average: {average_pos}")
    # logger.debug(f"Distance from centroid to average: {gemmi.Position(*average_pos).dist(gemmi.Position(*event_centroid))}")

    return transform


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


# def get_event_map(reference_xmap_grid, event: EventInterface, model: ModelInterface):
#     reference_xmap_grid_array = np.array(reference_xmap_grid, copy=True)
#
#     event_map_reference_grid = gemmi.FloatGrid(*[reference_xmap_grid.nu,
#                                                  reference_xmap_grid.nv,
#                                                  reference_xmap_grid.nw,
#                                                  ]
#                                                )
#     event_map_reference_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")  # xmap.xmap.spacegroup
#     event_map_reference_grid.set_unit_cell(reference_xmap_grid.unit_cell)
#
#     event_map_reference_grid_array = np.array(event_map_reference_grid,
#                                               copy=False,
#                                               )
#
#     mean_array = model.mean
#     event_map_reference_grid_array[:, :, :] = (reference_xmap_grid_array - (event.bdc.bdc * mean_array)) / (
#             1 - event.bdc.bdc)
#
#     return event_map_reference_grid

def save_dmap(dmap, path):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = dmap
    # if p1:
    #     ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    # else:
    #     ccp4.grid.symmetrize_max()
    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(str(path))

def sample_to_png(sample_point, array, reference_frame, dtag_array, mean_indexes, out_path):
    # sample_point = [0.9, -36.23, -59.64]
    samples = []
    for j in range(array.shape[0]):
        grid = reference_frame.unmask(SparseDMap(array[j, :].flatten()))
        sample = grid.interpolate_value(gemmi.Position(*sample_point))
        samples.append(sample)

    sample_array = np.array(samples)
    mean_samples_mask = np.zeros(dtag_array.size)
    mean_samples_mask[mean_indexes] = 1.0
    mean_samples = sample_array[mean_samples_mask == 1.0]
    other_samples = sample_array[mean_samples_mask == 0.0]
    sample_native = sample_array[0]

    plt.scatter(
        x=(np.zeros(mean_samples.size) + 1.0).flatten(),
        y=mean_samples,
        c='#1f77b4'
    )
    plt.scatter(
        x=(np.zeros(other_samples.size) + 2.0).flatten(),
        y=other_samples,
        c='#bcbd22'
    )
    plt.scatter(
        x=(np.zeros(sample_native.size) + 3.0).flatten(),
        y=sample_native,
        c='#d62728'
    )
    output_path = str(out_path)
    print(f"Saving to: {output_path}")
    plt.savefig(output_path)
    plt.clf()


def test_sparse_dmap_stream(data_dir, out_dir):
    print(f"Data dir is {data_dir} and output dir is {out_dir}")

    # Get processor
    processor = ProcessLocalRay(24)

    # Parse the FS
    print(f"##### Loading filesystem #####")
    fs: PanDDAFSInterface = PanDDAFS(Path(data_dir), Path(out_dir))

    # Get the datasets
    print(f"##### Getting datasets #####")
    datasets: Dict[str, DatasetInterface] = {
        dataset_dir.dtag: XRayDataset.from_paths(
            dataset_dir.input_pdb_file,
            dataset_dir.input_mtz_file,
            dataset_dir.input_ligands,
        )
        for dataset_dir
        in fs.input.dataset_dirs.values()
    }
    print(f"Got {len(datasets)} datasets")

    # Get the test dataset
    print(f"##### Getting test dataset #####")
    dtag_array = np.array(list(datasets.keys()))
    # dtag = "JMJD2DA-x390"
    dtag = "JMJD2DA-x427"

    for _j, _dtag in enumerate(dtag_array):
        print(_dtag)
        if _dtag == dtag:
            dtag_index =_j
    dtag = list(datasets.keys())[dtag_index]
    dataset = datasets[dtag]
    print(f"Test dataset is {dtag}")
    shutil.copy(dataset.structure.path, Path(out_dir) / "reference.pdb")
    shutil.copy(dataset.reflections.path, Path(out_dir) / "reference.mtz")


    print(f"##### Getting datasets in resolution #####")
    dataset_res = dataset.reflections.resolution() + 0.1
    res = max(dataset_res, list(sorted([_dataset.reflections.resolution() for _dataset in datasets.values()]))[60]+0.1)
    datasets_resolution = {_dtag: _dataset for _dtag, _dataset in datasets.items() if _dataset.reflections.resolution() < res}
    print(f"\tGot {len(datasets_resolution)} of dataset res {res} or higher!")

    dtag_array = np.array(list(datasets_resolution.keys()))
    for _j, _dtag in enumerate(dtag_array):
        print(_dtag)
        if _dtag == dtag:
            dtag_index =_j

    # Get the alignments
    print(f"##### Getting alignments #####")
    begin_align = time.time()
    # alignments: Dict[str, Alignment] = {_dtag: Alignment(datasets[_dtag], dataset) for _dtag in datasets}
    alignments: Dict[str, Alignment] = processor.process_dict(
        {_dtag: Partial(Alignment).paramaterise(datasets[_dtag], dataset) for _dtag in datasets_resolution}
    )
    finish_align = time.time()
    print(f"Got {len(alignments)} alignments in {round(finish_align - begin_align, 1)}")

    # Get the reference frame
    print(f"##### Getting reference frame #####")
    begin_get_frame = time.time()
    reference_frame: DFrame = DFrame(dataset, processor)
    finish_get_frame = time.time()
    print(f"Got reference frame in {round(finish_get_frame - begin_get_frame, 1)}")
    for resid, partition in reference_frame.partitioning.partitions.items():
        print(f"\tResid: {resid} : {partition.points.shape} {partition.positions[0,:]}")

    #
    grid = reference_frame.get_grid()
    grid_array = np.array(grid, copy=False)
    grid_array[reference_frame.mask.indicies] = 1.0
    save_dmap(
        grid,
        Path(out_dir) / f"reference.ccp4"
    )

    # Save a partition mask
    grid = reference_frame.get_grid()
    grid_array = np.array(grid, copy=False)
    for resid, partition in reference_frame.partitioning.partitions.items():
        # if int(resid.number) != 238:
        #     continue
        # print(partition.points)
        grid_array[
            (
                partition.points[:, 0].flatten(),
                partition.points[:, 1].flatten(),
                partition.points[:, 2].flatten(),
            )
        ] = 1.0
    save_dmap(
        grid,
        Path(out_dir) / f"reference_partitions.ccp4"
    )

    # exit()
    # Get the dmaps

    print(f"##### Getting sparse dmap loader #####")
    dmaps: SparseDMapStream = SparseDMapStream(
        datasets_resolution,
        reference_frame,
        alignments,
        [
            TruncateReflections(
                datasets_resolution,
                res,
            ),
            SmoothReflections(dataset)
        ],
    )

    # Load
    # print(f"##### Getting sparse xmaps #####")
    # time_begin = time.time()
    # dmaps_sparse: Dict[str, SparseDMap] = {
    #     dtag: dmaps.load(dtag)
    #     for dtag
    #     in datasets
    # }
    # time_finish = time.time()
    # print(f"Got sparse xmaps in {round(time_finish - time_begin, 1)}")
    #
    # print(f"##### Saving aligned maps #####")
    # time_begin = time.time()
    # for dtag, dmap_sparse in dmaps_sparse.items():
    #     save_dmap(
    #         reference_frame.unmask(dmap_sparse),
    #         Path(out_dir) / f"{dtag}.ccp4"
    #     )
    # time_finish = time.time()
    # print(f"Saved xmaps in {round(time_finish - time_begin, 1)}")

    print(f"##### Loading DMaps #####")
    # time_begin = time.time()
    # sparse_dmaps = {}
    # for dtag in datasets:
    #     print(f"##### {dtag} #####")
    #     sparse_dmaps[dtag] = dmaps.load(dtag)
    #     # save_dmap(
    #     #         reference_frame.unmask(dmap_sparse),
    #     #         Path(out_dir) / f"{dtag}.ccp4"
    #     #     )
    # time_finish = time.time()
    # print(f"Saved xmaps in {round(time_finish - time_begin, 1)}")
    # time_begin = time.time()
    # array = dmaps.array_load()
    # time_finish = time.time()
    # print(f"Loaded xmaps in {round(time_finish - time_begin, 1)} into shape {array.shape}")

    print(f"##### Loading DMaps parallel #####")
    # time_begin = time.time()
    # sparse_dmaps = {}
    # for dtag in datasets:
    #     print(f"##### {dtag} #####")
    #     sparse_dmaps[dtag] = dmaps.load(dtag)
    #     # save_dmap(
    #     #         reference_frame.unmask(dmap_sparse),
    #     #         Path(out_dir) / f"{dtag}.ccp4"
    #     #     )
    # time_finish = time.time()
    # print(f"Saved xmaps in {round(time_finish - time_begin, 1)}")
    time_begin = time.time()
    transforms_ref = processor.put(dmaps.transforms)
    reference_frame_ref = processor.put(reference_frame)
    dmaps_dict = processor.process_dict(
        {
            _dtag: Partial(SparseDMapStream.parallel_load).paramaterise(
            datasets_resolution[_dtag], alignments[_dtag], transforms_ref, reference_frame_ref
        )
            for _dtag
            in datasets_resolution}
    )
    time_finish = time.time()
    print(f"Parallel loaded xmaps in {round(time_finish - time_begin, 1)} into dict of length {len(dmaps_dict)}")
    array = np.vstack([dmap.data.reshape((1,-1) ) for dtag, dmap in dmaps_dict.items() ])

    print(f"##### Masking dmaps #####")
    # sparse_dmaps_inner = {}
    # for dtag in datasets:
    #     # print(f"##### {dtag} #####")
    #     sparse_dmaps_inner[dtag] = reference_frame.mask_inner(reference_frame.unmask(sparse_dmaps[dtag]))
    # sparse_dmap_inner_array = np.vstack([sparse_dmap_inner.data for sparse_dmap_inner in sparse_dmaps_inner.values()])
    time_begin = time.time()
    print([reference_frame.mask.indicies_sparse_inner[0].shape, len(reference_frame.mask.indicies_sparse_inner)])
    sparse_dmap_inner_array = array[:, reference_frame.mask.indicies_sparse_inner]
    time_finish = time.time()
    print(f"Masked in {round(time_finish - time_begin, 1)} with shape {sparse_dmap_inner_array.shape}")


    print(f"##### Pairwise distances #####")
    # sparse_dmaps_inner = {}
    # for dtag in datasets:
    #     # print(f"##### {dtag} #####")
    #     sparse_dmaps_inner[dtag] = reference_frame.mask_inner(reference_frame.unmask(sparse_dmaps[dtag]))
    # sparse_dmap_inner_array = np.vstack([sparse_dmap_inner.data for sparse_dmap_inner in sparse_dmaps_inner.values()])
    time_begin = time.time()
    # distances = spatial.distance.pdist(sparse_dmap_inner_array)
    pca = PCA(n_components=min(100, min(sparse_dmap_inner_array.shape)), svd_solver="randomized")
    transformed = pca.fit_transform(sparse_dmap_inner_array)
    time_finish = time.time()
    print(f"PCA'd in {round(time_finish - time_begin, 1)} with shape {transformed.shape}")

    print(f"##### Pairwise distances #####")
    # sparse_dmaps_inner = {}
    # for dtag in datasets:
    #     # print(f"##### {dtag} #####")
    #     sparse_dmaps_inner[dtag] = reference_frame.mask_inner(reference_frame.unmask(sparse_dmaps[dtag]))
    # sparse_dmap_inner_array = np.vstack([sparse_dmap_inner.data for sparse_dmap_inner in sparse_dmaps_inner.values()])
    time_begin = time.time()
    # distances = spatial.distance.pdist(sparse_dmap_inner_array)
    # pca = PCA(n_components=min(100, min(sparse_dmap_inner_array.shape)), svd_solver="randomized")
    distances = spatial.distance.squareform(spatial.distance.pdist(transformed))
    time_finish = time.time()
    print(f"Distance'd in {round(time_finish - time_begin, 1)} with shape {distances.shape}")

    from sklearn import mixture

    # distances = spatial.distance.pdist(sparse_dmap_inner_array)
    # pca = PCA(n_components=min(100, min(sparse_dmap_inner_array.shape)), svd_solver="randomized")
    # distances = spatial.distance.squareform(spatial.distance.pdist(transformed))
    # for j in range(20):
    #     print(f"######### {j} ############")
    #
    #     time_begin = time.time()
    #     clf = mixture.GaussianMixture(n_components=j+1, covariance_type="diag")
    #     predicted = clf.fit_predict(transformed)
    #     time_finish = time.time()
    #     predicted_classes, counts = np.unique(predicted, return_counts=True)
    #     print(f"\tFit-predicted in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    #     print(f"\tCounts are {counts}")
    #
    #     time_begin = time.time()
    #     bic = clf.bic(transformed)
    #     time_finish = time.time()
    #     print(f"\tBic in {round(time_finish - time_begin, 1)} with bic {bic}")
    #     # print(np.max(clf.predict_proba(transformed), axis=1))
    #
    #     print(f"Predicted: {predicted[0]}")
    #     for predicted_class in np.unique(predicted):
    #         cov_iv = np.diag(clf.precisions_[predicted_class, :].flatten())
    #         mean = clf.means_[predicted_class, :].flatten()
    #         distance = np.cbrt(spatial.distance.mahalanobis(transformed[0,:].flatten(), mean, cov_iv))
    #         print(f"\t\tDistance: {predicted_class} {distance}")
    #         # print(f"\t\t{clf.predict_proba(transformed)[0,:].flatten()}")

    # print(f"##### Kneighbours #####")
    # time_begin = time.time()
    # kdt = NearestNeighbors(n_neighbors=31, n_jobs=-1).fit(transformed)
    # distances, neighbours = kdt.kneighbors(transformed)
    # time_finish = time.time()
    # print(f"Got k nearest neighbours in {round(time_finish - time_begin, 1)}")
    #
    # print(f"##### Z maps #####")
    # dtag_array = np.array([_dtag for _dtag in datasets_resolution])
    # used_dtags = np.zeros(dtag_array.shape)
    # max_dists = np.max(distances, axis=1)
    # ordered_indexes = np.argsort(max_dists)
    # for index in ordered_indexes:
    #     dtag = dtag_array[index]
    #     neighbour_indexes = neighbours[index, :]
    #     num_used_indexes = np.sum(used_dtags[neighbour_indexes])
    #
    #     if num_used_indexes != 0:
    #         print(f"\t\tAlready used dtags, skipping!")
    #         continue
    #     else:
    #         print(f"\t\tModel based on dtag: {dtag}!")
    #         print(f"\t\tModel using dtags: {dtag_array[neighbour_indexes]}!")
    #
    #     used_dtags[neighbour_indexes] = 1
    #
    #     masked_array = array[neighbour_indexes, :]
    #     mean = np.mean(masked_array, axis=0)
    #     std = np.std(masked_array, axis=0)
    #
    #
    #     # Sample datasets at low point
    #     sample_point = [1.12,-41.6,-53.83]
    #     samples = []
    #     for j in range(array.shape[0]):
    #         grid = reference_frame.unmask(SparseDMap(array[j,:].flatten()))
    #         sample  = grid.interpolate_value(gemmi.Position(*sample_point))
    #         samples.append(sample)
    #
    #     sample_array = np.array(samples)
    #     mean_samples_mask = np.zeros(dtag_array.size)
    #     mean_samples_mask[neighbour_indexes] = 1.0
    #     mean_samples = sample_array[mean_samples_mask == 1.0]
    #     other_samples = sample_array[mean_samples_mask == 0.0]
    #     sample_native = sample_array[0]
    #
    #     plt.scatter(
    #         x=(np.zeros(mean_samples.size) + 1.0).flatten(),
    #         y=mean_samples,
    #         c='#1f77b4'
    #     )
    #     plt.scatter(
    #         x=(np.zeros(other_samples.size) + 2.0).flatten(),
    #         y=other_samples,
    #         c='#bcbd22'
    #     )
    #     plt.scatter(
    #         x=(np.zeros(sample_native.size) + 3.0).flatten(),
    #         y=sample_native,
    #         c='#d62728'
    #     )
    #     output_path = str(Path(out_dir) / "samples_low.png")
    #     print(f"Saving to: {output_path}")
    #     plt.savefig(output_path)
    #
    #     plt.clf()
    #     grid = reference_frame.unmask(SparseDMap(std.flatten()))
    #     sample  = grid.interpolate_value(gemmi.Position(*sample_point))
    #     print(f"STD at position low is: {sample}")
    #     grid = reference_frame.unmask(SparseDMap(mean.flatten()))
    #     sample  = grid.interpolate_value(gemmi.Position(*sample_point))
    #     print(f"Mean at position low is: {sample}")
    #
    #
    #     # Sample mean at high point
    #     sample_point = [1.11, -37.88, -58.43]
    #     samples = []
    #     for j in range(array.shape[0]):
    #         grid = reference_frame.unmask(SparseDMap(array[j, :].flatten()))
    #         sample = grid.interpolate_value(gemmi.Position(*sample_point))
    #         samples.append(sample)
    #
    #     sample_array = np.array(samples)
    #     mean_samples_mask = np.zeros(dtag_array.size)
    #     mean_samples_mask[neighbour_indexes] = 1.0
    #     mean_samples = sample_array[mean_samples_mask == 1.0]
    #     other_samples = sample_array[mean_samples_mask == 0.0]
    #     sample_native = sample_array[0]
    #
    #     plt.scatter(
    #         x=(np.zeros(mean_samples.size) + 1.0).flatten(),
    #         y=mean_samples,
    #         c='#1f77b4'
    #     )
    #     plt.scatter(
    #         x=(np.zeros(other_samples.size) + 2.0).flatten(),
    #         y=other_samples,
    #         c='#bcbd22'
    #     )
    #     plt.scatter(
    #         x=(np.zeros(sample_native.size) + 3.0).flatten(),
    #         y=sample_native,
    #         c='#d62728'
    #     )
    #     output_path = str(Path(out_dir) / "samples_high.png")
    #     print(f"Saving to: {output_path}")
    #     plt.savefig(output_path)
    #     plt.clf()
    #
    #     grid = reference_frame.unmask(SparseDMap(std.flatten()))
    #     sample  = grid.interpolate_value(gemmi.Position(*sample_point))
    #     print(f"STD at position high is: {sample}")
    #     grid = reference_frame.unmask(SparseDMap(mean.flatten()))
    #     sample  = grid.interpolate_value(gemmi.Position(*sample_point))
    #     print(f"Mean at position low is: {sample}")
    #
    #     # Sample mean at med point
    #     sample_point = [0.9,-36.23, -59.64]
    #     samples = []
    #     for j in range(array.shape[0]):
    #         grid = reference_frame.unmask(SparseDMap(array[j, :].flatten()))
    #         sample = grid.interpolate_value(gemmi.Position(*sample_point))
    #         samples.append(sample)
    #
    #     sample_array = np.array(samples)
    #     mean_samples_mask = np.zeros(dtag_array.size)
    #     mean_samples_mask[neighbour_indexes] = 1.0
    #     mean_samples = sample_array[mean_samples_mask == 1.0]
    #     other_samples = sample_array[mean_samples_mask == 0.0]
    #     sample_native = sample_array[0]
    #
    #     plt.scatter(
    #         x=(np.zeros(mean_samples.size) + 1.0).flatten(),
    #         y=mean_samples,
    #         c='#1f77b4'
    #     )
    #     plt.scatter(
    #         x=(np.zeros(other_samples.size) + 2.0).flatten(),
    #         y=other_samples,
    #         c='#bcbd22'
    #     )
    #     plt.scatter(
    #         x=(np.zeros(sample_native.size) + 3.0).flatten(),
    #         y=sample_native,
    #         c='#d62728'
    #     )
    #     output_path = str(Path(out_dir) / "samples_med.png")
    #     print(f"Saving to: {output_path}")
    #     plt.savefig(output_path)
    #     plt.clf()
    #
    #     grid = reference_frame.unmask(SparseDMap(std.flatten()))
    #     sample = grid.interpolate_value(gemmi.Position(*sample_point))
    #     print(f"STD at position high is: {sample}")
    #     grid = reference_frame.unmask(SparseDMap(mean.flatten()))
    #     sample  = grid.interpolate_value(gemmi.Position(*sample_point))
    #     print(f"Mean at position low is: {sample}")
    #
    #     dataset_grid = reference_frame.unmask(SparseDMap(array[0,:].flatten()))
    #     # dataset_grid.symmetrize_abs_max()
    #     dataset_reflections= Reflections.from_grid(dataset_grid,
    #                                                dataset,
    #                                                res
    #                                                )
    #
    #     mean_grid = reference_frame.unmask(SparseDMap(mean.flatten()))
    #     # mean_grid.symmetrize_abs_max()
    #     mean_reflections = Reflections.from_grid(mean_grid,
    #                                              dataset,
    #                                              res
    #                                              )
    #     # mean_grid = mean_reflections.transform_f_phi_to_map(exact_size=reference_frame.spacing)
    #     # mean_grid.symmetrize_abs_max()
    #
    #     # dataset_array = np.array(reference_frame.unmask(SparseDMap(array[0,:].flatten())), copy=False)
    #     # mean_array = np.array(reference_frame.unmask(SparseDMap(mean.flatten())), copy=False)
    #
    #     print(f"After undoing from grid shape {np.array(mean_grid).shape}")
    #     dataset_masked = XRayDataset(dataset.structure, dataset_reflections, dataset.ligand_files)
    #     mean_dataset_masked = XRayDataset(dataset.structure, mean_reflections, dataset.ligand_files)
    #     # mean_smoothed_grid = SmoothReflections(dataset_masked)(mean_dataset_masked).reflections.transform_f_phi_to_map(exact_size=reference_frame.spacing)
    #     mean_smoothed_grid = SmoothReflections(dataset_masked).real_space_smooth(mean_dataset_masked, reference_frame.mask.indicies_inner, reference_frame.spacing).reflections.transform_f_phi_to_map(exact_size=reference_frame.spacing)
    #     print(f"Mean smoothed shape: {np.array(mean_smoothed_grid).shape} vs original {np.array(reference_frame.unmask(SparseDMap(array[0,:].flatten()))).shape}")
    #     save_dmap(
    #         mean_smoothed_grid,
    #         Path(out_dir) / f"{dtag}_mean_smoothed.ccp4"
    #     )
    #
    #     plt.scatter(
    #         x=np.sort(array[dtag_index,:].flatten(), axis=None),
    #         y=np.sort(mean.flatten(), axis=None),
    #         c='#1f77b4'
    #     )
    #     plt.savefig(str(Path(out_dir) / "quantiles.png"))
    #     plt.clf()
    #     # from scipy.ndimage import gaussian_filter
    #     # from scipy.optimize import shgo
    #     #
    #     # from scipy.signal import convolve
    #     # def filter_gauss(sigma):
    #     #     return np.linalg.norm((dataset_array - gaussian_filter(mean_array, sigma=sigma)).flatten())
    #     #
    #     # def convolve_gauss(sigma):
    #     #
    #     #     np.linalg.norm((dataset_array - convolve(mean_array, vals.reshape((3,3,3)))).flatten())
    #     #
    #     # default = filter_gauss(np.array([0.0,0.0,0.0]))
    #     # blurred_10 = filter_gauss(np.array([10.0,10.0,10.0]))
    #     # blurred_1 = filter_gauss(np.array([1.0,1.0,1.0]))
    #     # blurred_01 = filter_gauss(np.array([0.1,0.1,0.1]))
    #     # bounds = [(0.00000001,100), (0.00000001,100), (0.00000001,100)]
    #     # res = shgo(filter_gauss, bounds)
    #     # print(res.x)
    #     # # print([res.fun, default, blurred])
    #     # print([default, blurred_01, blurred_1, blurred_10, res.fun])
    #     #
    #     # # mean_smoothed_array = gaussian_filter(mean_array, sigma=res.x)
    #     # save_dmap(
    #     #     reference_frame.unmask(SparseDMap(mean_smoothed_array[reference_frame.mask.indicies])),
    #     #     Path(out_dir) / f"{dtag}_mean_smoothed_gauss.ccp4"
    #     # )
    #     std = np.std(masked_array, axis=0)
    #     z = ((array[dtag_index,:]-mean) / std)
    #     normalized_z = z / np.std(z)
    #
    #     z_grid = reference_frame.unmask(SparseDMap(z))
    #
    #     normalized_z_grid = reference_frame.unmask(SparseDMap(normalized_z))
    #
    #     save_dmap(
    #         z_grid,
    #         Path(out_dir) / f"{dtag}_z.ccp4"
    #     )
    #
    #     low_z = np.zeros(z.shape)
    #     low_z[z < 2.0] = 1.0
    #     low_z_grid = reference_frame.unmask(SparseDMap(low_z))
    #     save_dmap(
    #         low_z_grid,
    #         Path(out_dir) / f"{dtag}_low_z.ccp4"
    #     )
    #     neg_z = np.zeros(z.shape)
    #     neg_z[z < -2.0] = 1.0
    #     neg_z_grid = reference_frame.unmask(SparseDMap(neg_z))
    #     save_dmap(
    #         neg_z_grid,
    #         Path(out_dir) / f"{dtag}_neg_z.ccp4"
    #     )
    #
    #     mean_grid = reference_frame.unmask(SparseDMap(mean))
    #     save_dmap(
    #         mean_grid,
    #         Path(out_dir) / f"{dtag}_mean.ccp4"
    #     )
    #
    #     std_grid = reference_frame.unmask(SparseDMap(std))
    #     save_dmap(
    #         std_grid,
    #         Path(out_dir) / f"{dtag}_std.ccp4"
    #     )
    #
    #     masked_mean = mean[reference_frame.mask.indicies_sparse_inner_atomic]
    #     print(f"Masked mean shape: {masked_mean.shape}")
    #     median = np.median(masked_mean)
    #     print(f"Median: {median}")
    #
    #     median_grid = reference_frame.unmask(SparseDMap((array[dtag_index,:]-mean) / (0.1*median)))
    #     save_dmap(
    #         median_grid,
    #         Path(out_dir) / f"{dtag}_median_diff.ccp4"
    #     )

    # for dtag, neighbour_indexes, dtag_dists in zip(datasets_resolution, neighbours, distances):
    #     neighbour_dtags = dtag_array[neighbour_indexes.flatten()]
    #     print(f"\t{dtag} : {neighbour_dtags[:3]} : {np.max(dtag_dists)}")

    print(f"##### Z maps #####")
    # for predicted_class, count in zip(predicted_classes, counts):
    #     if count < 20:
    #         continue
    #
    #     masked_array = array[predicted == predicted_class, :]
    #     mean = np.median(masked_array, axis=0)
    #     std = np.std(masked_array, axis=0)
    #     z = (array[0,:]-mean / std)
    #     normalized_z = z / np.std(z)
    #
    #     z_grid = reference_frame.unmask(SparseDMap(z))
    #
    #     save_dmap(
    #         z_grid,
    #         Path(out_dir) / f"{predicted_class}_z.ccp4"
    #     )
    #
    #     normalized_z_grid = reference_frame.unmask(SparseDMap(normalized_z))
    #
    #     save_dmap(
    #         normalized_z_grid,
    #         Path(out_dir) / f"{predicted_class}_normalized_z.ccp4"
    #     )
    #
    #     mean_grid = reference_frame.unmask(SparseDMap(mean))
    #     save_dmap(
    #         mean_grid,
    #         Path(out_dir) / f"{predicted_class}_mean.ccp4"
    #     )

    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="diag", weight_concentration_prior=100000.0)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    # ##############################
    #
    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="diag",
    #                                         weight_concentration_prior=1/100000.0)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    # ##############################
    #
    #
    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="full",
    #                                         weight_concentration_prior=100000.0)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class, ]
    #
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    # ##############################
    #
    #
    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="full",
    #                                         weight_concentration_prior=1 / 100000.0)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class, ]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    # ##############################


    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=30, covariance_type="diag",)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class, ]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    # print(f"Predicted: {predicted[1]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class,]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[1, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    # print(f"Predicted: {predicted[2]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class,]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[2, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    # print(f"Predicted: {predicted[3]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class,]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[3, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    #
    # for predicted_class, count in zip(predicted_classes, counts):
    #     if count < 20:
    #         continue
    #
    #     masked_array = array[predicted == predicted_class, :]
    #     mean = np.median(masked_array, axis=0)
    #     std = np.std(masked_array, axis=0)
    #     z = (array[0,:]-mean / std)
    #     normalized_z = z / np.std(z)
    #
    #     z_grid = reference_frame.unmask(SparseDMap(z))
    #
    #     save_dmap(
    #         z_grid,
    #         Path(out_dir) / f"{predicted_class}_z.ccp4"
    #     )
    #
    #     normalized_z_grid = reference_frame.unmask(SparseDMap(normalized_z))
    #
    #     save_dmap(
    #         normalized_z_grid,
    #         Path(out_dir) / f"{predicted_class}_normalized_z.ccp4"
    #     )
    #
    #     mean_grid = reference_frame.unmask(SparseDMap(mean))
    #     save_dmap(
    #         mean_grid,
    #         Path(out_dir) / f"{predicted_class}_mean.ccp4"
    #     )






        # print(f"\t\t{clf.predict_proba(transformed)[0,:].flatten()}")
    #
    time_begin = time.time()
    dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="diag")
    predicted = dpgmm.fit_predict(transformed)
    time_finish = time.time()
    print(f"\tFit-predicted bayesian full in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    predicted_classes, counts = np.unique(predicted, return_counts=True)
    for _dtag, prediction in zip(datasets, predicted):
        print(f"\t\t{_dtag} {prediction}")

    print(f"\tBayesian counts are {counts}")
    for predicted_class, count in zip(predicted_classes, counts):
        if count < 20:
            continue

        dtag_array = array[dtag_index,:]
        masked_array = array[predicted == predicted_class, :]
        mean = np.mean(masked_array, axis=0)
        std = np.std(masked_array, axis=0)
        z = ((array[dtag_index,:]-mean) / std)


        z_grid = reference_frame.unmask(SparseDMap(z))
        save_dmap(
            z_grid,
            Path(out_dir) / f"bayes_{predicted_class}_{dtag}_z.ccp4"
        )

        mean_grid = reference_frame.unmask(SparseDMap(mean))
        save_dmap(
            mean_grid,
            Path(out_dir) / f"bayes_{predicted_class}_{dtag}_mean.ccp4"
        )

        begin = time.time()
        point_array = np.vstack([partition.points for resid, partition in reference_frame.partitioning.partitions.items()]) % np.array(reference_frame.spacing)
        position_array = np.vstack([partition.positions for resid, partition in reference_frame.partitioning.partitions.items()])

        _all_points_array = point_array
        # all_points_array = _all_points_array - np.min(_all_points_array, axis=0).reshape((1, 3))
        all_points_array = point_array
        all_positions_array = position_array

        # print(f"All points shape: {all_points_array.shape}")
        # print(f"All positions shape: {all_positions_array.shape}")

        # unique_points, indexes = np.unique(all_points_array, axis=0, return_index=True)
        all_point_indexes = (all_points_array[:, 0], all_points_array[:, 1], all_points_array[:, 2],)
        # shape = (np.max(all_points_array, axis=0) - np.min(all_points_array, axis=0)) + 1
        shape = reference_frame.spacing
        point_3d_array_x = np.zeros((shape[0], shape[1], shape[2]),)
        point_3d_array_y = np.zeros((shape[0], shape[1], shape[2]),)
        point_3d_array_z = np.zeros((shape[0], shape[1], shape[2]),)

        point_3d_array_x[all_point_indexes] = all_points_array[:, 0]
        point_3d_array_y[all_point_indexes] = all_points_array[:, 1]
        point_3d_array_z[all_point_indexes] = all_points_array[:, 2]


        # point_3d_array[all_point_indexes] = True
        # initial_unique_points = np.argwhere(point_3d_array)
        # unique_points = initial_unique_points + np.min(_all_points_array, axis=0).reshape((1, 3))
        # unique_points_indexes = (initial_unique_points[:, 0], initial_unique_points[:, 1], initial_unique_points[:, 2],)
        pos_3d_arr_x = np.zeros((shape[0], shape[1], shape[2]))
        pos_3d_arr_y = np.zeros((shape[0], shape[1], shape[2]))
        pos_3d_arr_z = np.zeros((shape[0], shape[1], shape[2]))

        pos_3d_arr_x[all_point_indexes] = all_positions_array[:, 0]
        pos_3d_arr_y[all_point_indexes] = all_positions_array[:, 1]
        pos_3d_arr_z[all_point_indexes] = all_positions_array[:, 2]

        z_unmasked_array = np.array(z_grid, copy=False)

        high_z_indexes = np.nonzero(z_unmasked_array > 2.0)
        high_z_pos_x = pos_3d_arr_x[high_z_indexes]
        high_z_pos_y = pos_3d_arr_y[high_z_indexes]
        high_z_pos_z = pos_3d_arr_z[high_z_indexes]

        high_z_pos_array = np.hstack([
            high_z_pos_x.reshape((-1,1)),
            high_z_pos_y.reshape((-1, 1)),
            high_z_pos_z.reshape((-1, 1))
        ])

        high_z_point_x = point_3d_array_x[high_z_indexes]
        high_z_point_y = point_3d_array_y[high_z_indexes]
        high_z_point_z = point_3d_array_z[high_z_indexes]

        high_z_point_array = np.hstack([
            high_z_point_x.reshape((-1,1)),
            high_z_point_y.reshape((-1, 1)),
            high_z_point_z.reshape((-1, 1))
        ])


        finish = time.time()
        print(f"Got high z poss in {round(finish-begin, 1)}")
        print(high_z_pos_array)
        print(high_z_pos_array.shape)
        print([np.min(high_z_pos_array, axis=0), np.max(high_z_pos_array, axis=0)])


        print(high_z_point_array)
        print(high_z_point_array.shape)
        print([np.min(high_z_point_array, axis=0), np.max(high_z_point_array, axis=0)])
        print([np.min(point_array, axis=0), np.max(point_array, axis=0)])

        # high_z_points = np.hstack([
        #     high_z_indexes[0].reshape(-1,1),
        #     high_z_indexes[1].reshape(-1, 1),
        #     high_z_indexes[2].reshape(-1, 1),
        # ])

        from sklearn.cluster import DBSCAN
        clusters = DBSCAN(eps=1.0, min_samples=5).fit_predict(high_z_pos_array)

        cluster_nums, counts = np.unique(clusters, return_counts=True)



        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        print(f"Device: {dev}")

        # Load the model
        cnn = resnet18(num_classes=2, num_input=4)
        cnn_path = Path(os.path.dirname(inspect.getfile(resnet))) / "model.pt"
        cnn.load_state_dict(torch.load(cnn_path, map_location=dev))

        # Add model to device
        cnn.to(dev)
        cnn.eval()
        cnn = cnn.float()

        # Annotate the events
        event_scores = {}
        # for event_id, event in events.events.items():

        xmap_grid = reference_frame.unmask(SparseDMap(dtag_array))
        mean_grid = reference_frame.unmask(SparseDMap(mean))

        for cluster_num, count in zip(cluster_nums, counts):
            if cluster_num == -1:
                continue
            volume = count * (z_grid.unit_cell.volume / grid.point_count)
            if volume > 5.0:
                print(f"\tCluster: {cluster_num} : {np.mean(high_z_pos_array[clusters == cluster_num, :], axis=0)} : size: {count}: vol {volume}")

                centroid = np.mean(high_z_pos_array[clusters == cluster_num, :], axis=0)

                n = 30
                # sample_array = np.zeros((n, n, n), dtype=np.float32)
                sample_transform = get_sample_transform_from_event(
                    centroid,
                    0.5,
                    n,
                    3.5
                )
                sample_array = np.zeros((n, n, n), dtype=np.float32)

                bdcs = np.linspace(0.0,0.95,20).reshape((20,1,1,1))
                xmap_sample = sample_xmap(xmap_grid, sample_transform, np.copy(sample_array))
                mean_map_sample = sample_xmap(mean_grid, sample_transform, np.copy(sample_array))

                image_events = (xmap_sample[np.newaxis, :] - (bdcs * mean_map_sample[np.newaxis, :])) / (1-bdcs)
                print(f"Image evnets: {image_events.shape}")


                # event_map = get_event_map(dataset_xmap.xmap, event, model)
                # sample_array_event = np.copy(sample_array)
                # image_event = sample_xmap(event_map, sample_transform, sample_array_event)

                # sample_array_raw = np.copy(sample_array)
                # image_raw = sample_xmap(dataset_xmap.xmap, sample_transform, sample_array_raw)
                image_raw = np.stack([xmap_sample for _j in range(20)])

                sample_array_zmap = np.copy(sample_array)
                zmap_sample = sample_xmap(z_grid, sample_transform, sample_array_zmap)
                image_zmap = np.stack([zmap_sample for _j in range(20)])

                sample_array_model = np.copy(sample_array)
                model_map = get_model_map(dataset.structure.structure, xmap_grid)
                model_sample = sample_xmap(model_map, sample_transform, sample_array_model)
                image_model = np.stack([model_sample for _j in range(20)])

                image = np.stack([image_events, image_raw, image_zmap, image_model], axis=1)
                print([image.shape, image.dtype])

                # Transfer to tensor
                # image_t = torch.unsqueeze(torch.from_numpy(image), 0)
                image_t = torch.from_numpy(image)

                # Move tensors to device
                image_c = image_t.to(dev)

                # Run model
                model_annotation = cnn(image_c.float())

                # Track score
                model_annotations = model_annotation.to(torch.device("cpu")).detach().numpy()
                for _j in range(20):
                    bdc = bdcs.flatten()[_j]
                    annotation = model_annotations[_j, 1]
                    print(f"\t\t{np.round(bdc, 2)} {np.round(annotation, 2)}")


        for bdc in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            event_array = (dtag_array - (bdc*mean)) / (1-bdc)
            event_grid = reference_frame.unmask(SparseDMap(event_array))

            save_dmap(
                event_grid,
                Path(out_dir) / f"bayes_{predicted_class}_{dtag}_event_{round(bdc,1)}.ccp4"
            )

        # grid = reference_frame.unmask(SparseDMap(std.flatten()))

        # sample_point = [1.12,-41.6,-53.83]
        # out_path = Path(out_dir) / f"bayes_fig_{predicted_class}_low.png"
        # sample_to_png(sample_point, array, reference_frame, dtag_array, predicted == predicted_class, out_path)
        # sample = grid.interpolate_value(gemmi.Position(*sample_point))
        # print(f"STD at position low is: {sample}")
        #
        # sample_point = [1.11, -37.88, -58.43]
        # out_path = Path(out_dir) / f"bayes_fig_{predicted_class}_high.png"
        # sample_to_png(sample_point, array, reference_frame, dtag_array, predicted == predicted_class, out_path)
        # sample = grid.interpolate_value(gemmi.Position(*sample_point))
        # print(f"STD at position high is: {sample}")
        #
        # sample_point = [0.9, -36.23, -59.64]
        # out_path = Path(out_dir) / f"bayes_fig_{predicted_class}_med.png"
        # sample_to_png(sample_point, array, reference_frame, dtag_array, predicted == predicted_class, out_path)
        # sample = grid.interpolate_value(gemmi.Position(*sample_point))
        # print(f"STD at position med is: {sample}")

    print(f"##### Z maps #####")
    # for predicted_class, count in zip(predicted_classes, counts):
    #     if count < 20:
    #         continue
    #
    #     masked_array = array[predicted == predicted_class, :]
    #     mean = np.median(masked_array, axis=0)
    #     std = np.std(masked_array, axis=0)
    #     z = (array[0,:]-mean / std)
    #     normalized_z = z / np.std(z)
    #
    #     z_grid = reference_frame.unmask(SparseDMap(z))
    #
    #     save_dmap(
    #         z_grid,
    #         Path(out_dir) / f"{predicted_class}_z.ccp4"
    #     )
    #
    #     normalized_z_grid = reference_frame.unmask(SparseDMap(normalized_z))
    #
    #     save_dmap(
    #         normalized_z_grid,
    #         Path(out_dir) / f"{predicted_class}_normalized_z.ccp4"
    #     )
    #
    #     mean_grid = reference_frame.unmask(SparseDMap(mean))
    #     save_dmap(
    #         mean_grid,
    #         Path(out_dir) / f"{predicted_class}_mean.ccp4"
    #     )

    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="diag", weight_concentration_prior=100000.0)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    # ##############################
    #
    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="diag",
    #                                         weight_concentration_prior=1/100000.0)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    # ##############################
    #
    #
    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="full",
    #                                         weight_concentration_prior=100000.0)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class, ]
    #
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    # ##############################
    #
    #
    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="full",
    #                                         weight_concentration_prior=1 / 100000.0)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class, ]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    # ##############################

    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=30, covariance_type="diag",)
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    #
    # print(f"Predicted: {predicted[0]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class, ]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[0, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    # print(f"Predicted: {predicted[1]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class,]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[1, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    # print(f"Predicted: {predicted[2]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class,]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[2, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    # print(f"Predicted: {predicted[3]}")
    # for predicted_class in np.unique(predicted):
    #     # cov_iv = np.diag(dpgmm.precisions_[predicted_class, :].flatten())
    #     cov_iv = dpgmm.precisions_[predicted_class,]
    #     mean = dpgmm.means_[predicted_class, :].flatten()
    #     distance = spatial.distance.mahalanobis(transformed[3, :].flatten(), mean, cov_iv)
    #     print(f"\t\tDistance: {predicted_class} {distance}")
    #
    #
    # for predicted_class, count in zip(predicted_classes, counts):
    #     if count < 20:
    #         continue
    #
    #     masked_array = array[predicted == predicted_class, :]
    #     mean = np.median(masked_array, axis=0)
    #     std = np.std(masked_array, axis=0)
    #     z = (array[0,:]-mean / std)
    #     normalized_z = z / np.std(z)
    #
    #     z_grid = reference_frame.unmask(SparseDMap(z))
    #
    #     save_dmap(
    #         z_grid,
    #         Path(out_dir) / f"{predicted_class}_z.ccp4"
    #     )
    #
    #     normalized_z_grid = reference_frame.unmask(SparseDMap(normalized_z))
    #
    #     save_dmap(
    #         normalized_z_grid,
    #         Path(out_dir) / f"{predicted_class}_normalized_z.ccp4"
    #     )
    #
    #     mean_grid = reference_frame.unmask(SparseDMap(mean))
    #     save_dmap(
    #         mean_grid,
    #         Path(out_dir) / f"{predicted_class}_mean.ccp4"
    #     )

    # print(f"\t\t{clf.predict_proba(transformed)[0,:].flatten()}")
    #

    #
    # time_begin = time.time()
    # dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="tied")
    # predicted = dpgmm.fit_predict(transformed)
    # time_finish = time.time()
    # print(f"\tFit-predicted bayesian tied in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    # predicted_classes, counts = np.unique(predicted, return_counts=True)
    # for dtag, prediction in zip(datasets, predicted):
    #     print(f"\t\t{dtag} {prediction}")
    #
    # print(f"\tBayesian counts are {counts}")
    # print(np.max(dpgmm.predict_proba(transformed), axis=1))


if __name__ == "__main__":
    fire.Fire(test_sparse_dmap_stream)
