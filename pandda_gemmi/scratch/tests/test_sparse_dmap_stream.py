import fire

from typing import Dict, List
import time
from pathlib import Path

import gemmi
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA

from pandda_gemmi.scratch.interfaces import *
from pandda_gemmi.scratch.fs import PanDDAFS
from pandda_gemmi.scratch.dataset import XRayDataset
from pandda_gemmi.scratch.dmaps import DMap, SparseDMap, SparseDMapStream, TruncateReflections, SmoothReflections
from pandda_gemmi.scratch.alignment import Alignment, DFrame
from pandda_gemmi.scratch.processor import ProcessLocalRay, Partial

def save_dmap(dmap, path):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = dmap
    # if p1:
    #     ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    # else:
    #     ccp4.grid.symmetrize_max()
    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(str(path))


def test_sparse_dmap_stream(data_dir, out_dir):
    print(f"Data dir is {data_dir} and output dir is {out_dir}")

    # Get processor
    processor = ProcessLocalRay(12)

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
    dtag = list(datasets.keys())[0]
    dataset = datasets[dtag]
    print(f"Test dataset is {dtag}")

    # Get the alignments
    print(f"##### Getting alignments #####")
    begin_align = time.time()
    # alignments: Dict[str, Alignment] = {_dtag: Alignment(datasets[_dtag], dataset) for _dtag in datasets}
    alignments: Dict[str, Alignment] = processor.process_dict(
        {_dtag: Partial(Alignment).paramaterise(datasets[_dtag], dataset) for _dtag in datasets}
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
        datasets,
        reference_frame,
        alignments,
        [
            TruncateReflections(
                datasets,
                dataset.reflections.resolution(),
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
    time_begin = time.time()
    array = dmaps.array_load()
    time_finish = time.time()
    print(f"Loaded xmaps in {round(time_finish - time_begin, 1)} into shape {array.shape}")
    #
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
    for j in range(20):
        print(f"######### {j} ############")

        time_begin = time.time()
        clf = mixture.GaussianMixture(n_components=j+1, covariance_type="diag")
        predicted = clf.fit_predict(transformed)
        time_finish = time.time()
        predicted_classes, counts = np.unique(predicted, return_counts=True)
        print(f"\tFit-predicted in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
        print(f"\tCounts are {counts}")

        time_begin = time.time()
        bic = clf.bic(transformed)
        time_finish = time.time()
        print(f"\tBic in {round(time_finish - time_begin, 1)} with bic {bic}")
        # print(np.max(clf.predict_proba(transformed), axis=1))

        for predicted_class in np.unique(predicted):
            cov_iv = np.diag(clf.precisions_[predicted_class, :].flatten())
            mean = clf.means_[predicted_class, :].flatten()
            distance = np.cbrt(spatial.distance.mahalanobis(transformed[0,:].flatten(), mean, cov_iv))
            print(f"\t\tDistance: {predicted_class} {distance}")
            print(f"\t\t{clf.predict_proba(transformed)[0,:].flatten()}")

    time_begin = time.time()
    dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="diag")
    predicted = dpgmm.fit_predict(transformed)
    time_finish = time.time()
    print(f"\tFit-predicted bayesian in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    predicted_classes, counts = np.unique(predicted, return_counts=True)
    for dtag, prediction in zip(datasets, predicted):
        print(f"\t\t{dtag} {prediction}")

    print(f"\tBayesian counts are {counts}")

    time_begin = time.time()
    dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="full")
    predicted = dpgmm.fit_predict(transformed)
    time_finish = time.time()
    print(f"\tFit-predicted bayesian full in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    predicted_classes, counts = np.unique(predicted, return_counts=True)
    for dtag, prediction in zip(datasets, predicted):
        print(f"\t\t{dtag} {prediction}")

    print(f"\tBayesian counts are {counts}")

    time_begin = time.time()
    dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type="tied")
    predicted = dpgmm.fit_predict(transformed)
    time_finish = time.time()
    print(f"\tFit-predicted bayesian tied in {round(time_finish - time_begin, 1)} with shape {predicted.shape}")
    predicted_classes, counts = np.unique(predicted, return_counts=True)
    for dtag, prediction in zip(datasets, predicted):
        print(f"\t\t{dtag} {prediction}")

    print(f"\tBayesian counts are {counts}")
    print(np.max(dpgmm.predict_proba(transformed), axis=1))


if __name__ == "__main__":
    fire.Fire(test_sparse_dmap_stream)
