import time

import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors

from ..interfaces import *


def get_characterization_sets(
        dtag: str,
        datasets: Dict[str, DatasetInterface],
        dmaps: np.array,
        reference_frame: DFrameInterface,
        characterization_model,
        min_size: int = 15
):
    # Get the array of dataset dtags
    dtag_array = np.array([_dtag for _dtag in datasets])

    # Get the
    classes = characterization_model(dmaps, reference_frame)

    # Get the characterization sets from dmap mask
    characterization_sets = {}

    # Get the clusters and their membership numbers
    unique_classes, counts = np.unique(classes, return_counts=True)
    # print(f"Unique classes : {unique_classes} : counts : {counts}")
    j = 0
    for unique_class, count in zip(unique_classes, counts):
        if unique_class < 1:
            continue
        if count >= min_size:
            class_dtags = dtag_array[classes == unique_class]
            characterization_sets[j] = [str(_dtag) for _dtag in class_dtags]
            j = j + 1

    return characterization_sets


class CharacterizationGaussianMixture:
    def __init__(self, n_components=20, covariance_type="diag", ):
        self.n_components = n_components
        self.covariance_type = covariance_type

    def __call__(self, dmaps, reference_frame):
        # Get the inner mask of the density
        sparse_dmap_inner_array = dmaps[:, reference_frame.mask.indicies_sparse_inner]

        # Transform the data to a reasonable size for a GMM
        pca = PCA(n_components=min(100, min(sparse_dmap_inner_array.shape)), svd_solver="randomized")
        transformed = pca.fit_transform(sparse_dmap_inner_array)

        # Fit the Dirichlet Process Gaussian Mixture Model and predict component membership
        dpgmm = mixture.BayesianGaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type)
        predicted = dpgmm.fit_predict(transformed)

        return predicted

class CharacterizationNN:
    def __init__(self, n_neighbours=25, ):
        self.n_neighbours = n_neighbours

    def __call__(self, dmaps, reference_frame):
        # Get the inner mask of the density
        sparse_dmap_inner_array = dmaps[:, reference_frame.mask.indicies_sparse_inner]

        time_begin_fit = time.time()
        # # Transform the data to a reasonable size for a GMM
        pca = PCA(n_components=min(100, min(sparse_dmap_inner_array.shape)), svd_solver="randomized")
        transformed = pca.fit_transform(sparse_dmap_inner_array)

        # # Fit the Dirichlet Process Gaussian Mixture Model and predict component membership
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbours).fit(transformed)
        distances, indices = nbrs.kneighbors(transformed)
        time_finish_fit = time.time()
        # print(f"Nearest neighbours fit on pca dimension in time: {time_finish_fit - time_begin_fit}")

        # time_begin_fit = time.time()
        # # # Transform the data to a reasonable size for a GMM
        # pca = PCA(n_components=min(100, min(sparse_dmap_inner_array.shape)), svd_solver="arpack")
        # transformed = pca.fit_transform(sparse_dmap_inner_array)
        #
        # # # Fit the Dirichlet Process Gaussian Mixture Model and predict component membership
        # nbrs = NearestNeighbors(n_neighbors=self.n_neighbours).fit(transformed)
        # distances, indices = nbrs.kneighbors(transformed)
        # time_finish_fit = time.time()
        # print(f"Nearest neighbours fit on arpack pca dimension in time: {time_finish_fit - time_begin_fit}")
        #
        # time_begin_fit = time.time()
        # # # Transform the data to a reasonable size for a GMM
        # pca = IncrementalPCA(n_components=min(100, min(sparse_dmap_inner_array.shape)), batch_size=100)
        # transformed = pca.fit_transform(sparse_dmap_inner_array)
        #
        # # # Fit the Dirichlet Process Gaussian Mixture Model and predict component membership
        # nbrs = NearestNeighbors(n_neighbors=self.n_neighbours).fit(transformed)
        # distances, indices = nbrs.kneighbors(transformed)
        # time_finish_fit = time.time()
        # print(f"Nearest neighbours fit on ipca dimension in time: {time_finish_fit - time_begin_fit}")

        # Fit the Dirichlet Process Gaussian Mixture Model and predict component membership
        # time_begin_fit = time.time()
        # nbrs = NearestNeighbors(n_neighbors=self.n_neighbours, n_jobs=-1).fit(sparse_dmap_inner_array)
        # distances, indices = nbrs.kneighbors(sparse_dmap_inner_array)
        # time_finish_fit = time.time()
        # print(f"Nearest neighbours fit on full dimension in time: {time_finish_fit-time_begin_fit}")


        # Get neighbourhood radii
        radii = {}
        for j, row in enumerate(distances):
            mean_distance = np.mean(row)
            radii[j] = mean_distance

        # Sort datasets by radii
        radii_sorted = {index: radii[index] for index in sorted(radii, key=lambda _index: radii[_index])}

        # Loop over datasets from narrowest to broadest, checking whether any of their neighbours have been claimed
        # If so, skip to next
        claimed = []
        cluster_num = 1
        # cluster_widths = {}
        predicted = np.zeros(dmaps.shape[0], dtype=np.int)
        for index in radii_sorted:
            nearest_neighbour_indexes = indices[index, :]

            if not any([(j in claimed) for j in nearest_neighbour_indexes]):
                # CLaim indicies
                for j in nearest_neighbour_indexes:
                    claimed.append(j)

                predicted[nearest_neighbour_indexes] = cluster_num

                # cluster_widths[cluster_num] = radii_sorted[index]

                cluster_num += 1

        return predicted
