import time

from ..interfaces import *

from ..dataset import Reflections, XRayDataset
from .truncate_reflections import truncate_reflections, common_reflections

import numpy as np
import pandas as pd
import gemmi
from sklearn import neighbors
from scipy import optimize
from matplotlib import pyplot as plt

def get_rmsd(scale, y, r, y_inds, y_inds_unique, x_f):
    y_s = y * np.exp(scale * r)
        # knn_y = neighbors.RadiusNeighborsRegressor(0.01)
        # knn_y.fit(r.reshape(-1, 1),
        #           y_s.reshape(-1, 1),
        #           )
        #
        # y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

        # y_f = np.array(
        #     [np.mean(y_s[y_neighbours[1][j]]) for j, val in enumerate(sample_grid[:, np.newaxis].flatten())])

    y_f = np.array([np.mean(y_s[y_inds == rb]) for rb in y_inds_unique[1:-2]])

    _rmsd = np.sum(np.abs(x_f - y_f))
    return _rmsd

def get_rmsd_real_space(scale, reference_values, y, r, grid_mask, original_reflections, exact_size):
    original_reflections_array = np.array(original_reflections.reflections,
                                          copy=True,
                                          )

    original_reflections_table = pd.DataFrame(original_reflections_array,
                                              columns=original_reflections.reflections.column_labels(),
                                              )

    # f_array = original_reflections_table[original_reflections.f]

    original_reflections_table[original_reflections.f] = y * np.exp(scale * r)

    # New reflections
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = original_reflections.reflections.spacegroup
    new_reflections.set_cell_for_all(original_reflections.reflections.cell)

    # Add dataset
    new_reflections.add_dataset("scaled")

    # Add columns
    for column in original_reflections.reflections.columns:
        new_reflections.add_column(column.label, column.type)

    # Update
    new_reflections.set_data(original_reflections_table.to_numpy())

    # Update resolution
    new_reflections.update_reso()

    new_reflections_obj = Reflections(
        original_reflections.path,
        original_reflections.f,
        original_reflections.phi,
        new_reflections
    )

    new_grid = new_reflections_obj.transform_f_phi_to_map(exact_size=exact_size)
    new_grid_array = np.array(new_grid, copy=False)
    masked_new_grid_values = new_grid_array[grid_mask]


    _rmsd = np.linalg.norm(reference_values.flatten() - masked_new_grid_values.flatten())
    print(f"\t\tScale: {scale} : RMSD: {_rmsd}")
    return _rmsd

class SmoothReflections:
    def __init__(self, dataset: DatasetInterface):
        self.reference_dataset = dataset

    def __call__dep(self, dataset: DatasetInterface):

        begin_smooth_reflections = time.time()


        # # Get common set of reflections
        common_reflections_set = common_reflections({"reference" : self.reference_dataset, "dtag": dataset})

        # # Truncate
        reference_reflections = truncate_reflections( self.reference_dataset.reflections.reflections, common_reflections_set) #.truncate_reflections(common_reflections)
        dtag_reflections = truncate_reflections( dataset.reflections.reflections, common_reflections_set) #truncate_reflections(common_reflections)

        # Refference array
        # reference_reflections = truncated_reference.reflections.reflections
        reference_reflections_array = np.array(reference_reflections,
                                               copy=True,
                                               )
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                                   columns=reference_reflections.column_labels(),
                                                   )
        reference_f_array = reference_reflections_table[self.reference_dataset.reflections.f].to_numpy()

        # Dtag array
        # dtag_reflections = truncated_dataset.reflections.reflections
        dtag_reflections_array = np.array(dtag_reflections,
                                          copy=True,
                                          )
        dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                              columns=dtag_reflections.column_labels(),
                                              )
        dtag_f_array = dtag_reflections_table[dataset.reflections.f].to_numpy()

        # Resolution array
        reference_resolution_array = reference_reflections.make_1_d2_array()
        dtag_resolution_array = dtag_reflections.make_1_d2_array()

        # Prepare optimisation
        x = reference_f_array
        y = dtag_f_array

        r = reference_resolution_array

        sample_grid = np.linspace(min(r), max(r), 100)

        knn_x = neighbors.RadiusNeighborsRegressor(0.01)
        knn_x.fit(r.reshape(-1, 1),
                  x.reshape(-1, 1),
                  )
        x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)

        scales = []
        rmsds = []

        knn_y = neighbors.RadiusNeighborsRegressor(0.01)
        knn_y.fit(r.reshape(-1, 1),
                  (y * np.exp(0.0 * r)).reshape(-1, 1),
                  )

        # y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

        y_neighbours = knn_y.radius_neighbors(sample_grid[:, np.newaxis])

        # Optimise the scale factor
        for scale in np.linspace(-15, 15, 300):
            y_s = y * np.exp(scale * r)
            # knn_y = neighbors.RadiusNeighborsRegressor(0.01)
            # knn_y.fit(r.reshape(-1, 1),
            #           y_s.reshape(-1, 1),
            #           )
            #
            # y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

            y_f = np.array(
                [np.mean(y_s[y_neighbours[1][j]]) for j, val in enumerate(sample_grid[:, np.newaxis].flatten())])

            _rmsd = np.sum(np.abs(x_f - y_f))

            scales.append(scale)
            rmsds.append(_rmsd)

        # x = reference_f_array
        # y = dtag_f_array
        #
        # x_r = reference_resolution_array
        # y_r = dtag_resolution_array
        #
        # min_r = max([min(x_r), min(y_r)])
        # max_r = min([max(x_r), max(y_r)])
        # r_bins = np.linspace(min_r, max_r, 100)
        # # y_r_bins = np.linspace(min(y_r), max(y_r), 100)
        # x_inds = np.digitize(r, r_bins,)
        # y_inds = np.digitize(r, r_bins, )
        #
        min_scale = scales[np.argmin(rmsds)]

        # Get the original reflections
        original_reflections = dataset.reflections.reflections

        original_reflections_array = np.array(original_reflections,
                                              copy=True,
                                              )

        original_reflections_table = pd.DataFrame(original_reflections_array,
                                                  columns=reference_reflections.column_labels(),
                                                  )

        f_array = original_reflections_table[dataset.reflections.f]

        f_scaled_array = f_array * np.exp(min_scale * original_reflections.make_1_d2_array())

        original_reflections_table[dataset.reflections.f] = f_scaled_array

        # New reflections
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = original_reflections.spacegroup
        new_reflections.set_cell_for_all(original_reflections.cell)

        # Add dataset
        new_reflections.add_dataset("scaled")

        # Add columns
        for column in original_reflections.columns:
            new_reflections.add_column(column.label, column.type)

        # Update
        new_reflections.set_data(original_reflections_table.to_numpy())

        # Update resolution
        new_reflections.update_reso()

        # Create new dataset
        smoothed_dataset = XRayDataset(
            dataset.structure,
            Reflections(dataset.reflections.path,
                        dataset.reflections.f,
                        dataset.reflections.phi,
                        new_reflections),
            dataset.ligand_files
            # smoothing_factor=float(min_scale)
        )

        finish_smooth_reflections = time.time()
        print(f"\t\tSmooth: {finish_smooth_reflections-begin_smooth_reflections}")

        return smoothed_dataset

    def __call__(self, dataset: DatasetInterface):

        begin_smooth_reflections = time.time()

        # # Get common set of reflections
        begin_common = time.time()
        common_reflections_set = common_reflections({"reference" : self.reference_dataset, "dtag": dataset})
        finish_common = time.time()
        print(f"\t\t\tCommon: {finish_common-begin_common} with shape {common_reflections_set.shape}")

        # # Truncate
        begin_truncate = time.time()
        reference_reflections = truncate_reflections(
            self.reference_dataset.reflections.reflections, common_reflections_set)
        dtag_reflections = truncate_reflections(
            dataset.reflections.reflections, common_reflections_set)
        finish_truncate = time.time()
        print(f"\t\t\tTruncate: {finish_truncate-begin_truncate}")

        # Refference array
        # reference_reflections = truncated_reference.reflections.reflections
        begin_preprocess = time.time()
        reference_reflections_array = np.array(reference_reflections,
                                               copy=True,
                                               )
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                                   columns=reference_reflections.column_labels(),
                                                   )
        reference_f_array = reference_reflections_table[self.reference_dataset.reflections.f].to_numpy()


        # Dtag array
        # dtag_reflections = truncated_dataset.reflections.reflections
        dtag_reflections_array = np.array(dtag_reflections,
                                          copy=True,
                                          )
        dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                              columns=dtag_reflections.column_labels(),
                                              )
        dtag_f_array = dtag_reflections_table[dataset.reflections.f].to_numpy()
        print(f"\t\t\tReference f array size: {reference_f_array.shape} and dtag f array size: {dtag_f_array.shape}")



        # Resolution array
        reference_resolution_array = reference_reflections.make_1_d2_array()
        dtag_resolution_array = dtag_reflections.make_1_d2_array()

        # Prepare optimisation
        x = reference_f_array
        y = dtag_f_array

        r = reference_resolution_array

        # ####################### NEW 20 #########################
        #
        # # Get the resolution bins
        # sample_grid = np.linspace(np.min(r), np.max(r), 20)
        #
        # # Get the array that maps x values to bins
        # x_inds = np.digitize(reference_resolution_array, sample_grid)
        #
        # # Get the bin averages
        # populated_bins, counts = np.unique(x_inds, return_counts=True)
        # x_f = np.array([np.mean(x[x_inds == rb]) for rb in populated_bins[1:-2]])
        # # print(f"\t\t\tsample NEW: {x_f}")
        # # print(f"\t\t\txf NEW: {x_f}")
        #
        # y_inds = np.digitize(dtag_resolution_array, sample_grid)
        #
        # finish_preprocess = time.time()
        # print(f"\t\t\tPreprocess: {finish_preprocess - begin_preprocess}")
        #
        # # Optimise the scale factor
        #
        # begin_solve = time.time()
        # # y_inds_unique = np.unique(y_inds)
        # min_scale = optimize.minimize(
        #     lambda _scale: get_rmsd(_scale, y, r, y_inds, populated_bins, x_f),
        #     0.0,
        # ).x
        #
        # # min_scale = optimize.fsolve(
        # #     lambda _scale: rmsd(_scale, y, r, y_inds, sample_grid, x_f),
        # #     0.0
        # # )
        # finish_solve = time.time()
        # print(f"\t\t\tSolve NEW 20: {finish_solve - begin_solve} with scale: {min_scale}")
        #
        # ####################### NEW 100 #########################
        #
        # # Get the resolution bins
        # sample_grid = np.linspace(np.min(r), np.max(r), 100)
        #
        # # Get the array that maps x values to bins
        # x_inds = np.digitize(reference_resolution_array, sample_grid)
        #
        # # Get the bin averages
        # populated_bins, counts = np.unique(x_inds, return_counts=True)
        # x_f = np.array([np.mean(x[x_inds == rb]) for rb in populated_bins[1:-2]])
        # # print(f"\t\t\tsample NEW: {x_f}")
        # # print(f"\t\t\txf NEW: {x_f}")
        #
        # y_inds = np.digitize(dtag_resolution_array, sample_grid)
        #
        # finish_preprocess = time.time()
        # print(f"\t\t\tPreprocess: {finish_preprocess - begin_preprocess}")
        #
        # # Optimise the scale factor
        #
        # begin_solve = time.time()
        # # y_inds_unique = np.unique(y_inds)
        # min_scale = optimize.minimize(
        #     lambda _scale: get_rmsd(_scale, y, r, y_inds, populated_bins, x_f),
        #     0.0,
        # ).x
        #
        # # min_scale = optimize.fsolve(
        # #     lambda _scale: rmsd(_scale, y, r, y_inds, sample_grid, x_f),
        # #     0.0
        # # )
        # finish_solve = time.time()
        # print(f"\t\t\tSolve NEW 100: {finish_solve - begin_solve} with scale: {min_scale}")



        ####################### NEW 20 bounded #########################

        # Get the resolution bins
        sample_grid = np.linspace(np.min(r), np.max(r), 20)

        # Get the array that maps x values to bins
        x_inds = np.digitize(reference_resolution_array, sample_grid)

        # Get the bin averages
        populated_bins, counts = np.unique(x_inds, return_counts=True)
        x_f = np.array([np.mean(x[x_inds == rb]) for rb in populated_bins[1:-2]])
        # print(f"\t\t\tsample NEW: {x_f}")
        # print(f"\t\t\txf NEW: {x_f}")

        y_inds = np.digitize(dtag_resolution_array, sample_grid)

        finish_preprocess = time.time()
        print(f"\t\t\tPreprocess: {finish_preprocess - begin_preprocess}")

        # Optimise the scale factor

        begin_solve = time.time()
        # y_inds_unique = np.unique(y_inds)
        min_scale = optimize.minimize(
            lambda _scale: get_rmsd(_scale, y, r, y_inds, populated_bins, x_f),
            0.0,
            bounds=((-15.0, 15.0),),
            tol=0.1
        ).x

        # min_scale = optimize.fsolve(
        #     lambda _scale: rmsd(_scale, y, r, y_inds, sample_grid, x_f),
        #     0.0
        # )
        finish_solve = time.time()
        print(f"\t\t\tSolve NEW BOUNDED 20: {finish_solve - begin_solve} with scale: {min_scale}")

        # ####################### NEW 100 bounded #########################
        #
        # # Get the resolution bins
        # sample_grid = np.linspace(np.min(r), np.max(r), 100)
        #
        # # Get the array that maps x values to bins
        # x_inds = np.digitize(reference_resolution_array, sample_grid)
        #
        # # Get the bin averages
        # populated_bins, counts = np.unique(x_inds, return_counts=True)
        # x_f = np.array([np.mean(x[x_inds == rb]) for rb in populated_bins[1:-2]])
        # # print(f"\t\t\tsample NEW: {x_f}")
        # # print(f"\t\t\txf NEW: {x_f}")
        #
        # y_inds = np.digitize(dtag_resolution_array, sample_grid)
        #
        # finish_preprocess = time.time()
        # print(f"\t\t\tPreprocess: {finish_preprocess - begin_preprocess}")
        #
        # # Optimise the scale factor
        #
        # begin_solve = time.time()
        # # y_inds_unique = np.unique(y_inds)
        # min_scale = optimize.minimize(
        #     lambda _scale: get_rmsd(_scale, y, r, y_inds, populated_bins, x_f),
        #     0.0,
        #     bounds=((-15.0, 15.0),)
        # ).x
        #
        # # min_scale = optimize.fsolve(
        # #     lambda _scale: rmsd(_scale, y, r, y_inds, sample_grid, x_f),
        # #     0.0
        # # )
        # finish_solve = time.time()
        # print(f"\t\t\tSolve NEW BOUNDED 100: {finish_solve - begin_solve} with scale: {min_scale}")
        #
        # ####################### NEW 20 SHGO #########################
        #
        # # Get the resolution bins
        # sample_grid = np.linspace(np.min(r), np.max(r), 20)
        #
        # # Get the array that maps x values to bins
        # x_inds = np.digitize(reference_resolution_array, sample_grid)
        #
        # # Get the bin averages
        # populated_bins, counts = np.unique(x_inds, return_counts=True)
        # x_f = np.array([np.mean(x[x_inds == rb]) for rb in populated_bins[1:-2]])
        #
        # y_inds = np.digitize(dtag_resolution_array, sample_grid)
        #
        # finish_preprocess = time.time()
        # print(f"\t\t\tPreprocess: {finish_preprocess - begin_preprocess}")
        #
        # # Optimise the scale factor
        #
        # begin_solve = time.time()
        # # y_inds_unique = np.unique(y_inds)
        # min_scale = optimize.shgo(
        #     lambda _scale: get_rmsd(_scale, y, r, y_inds, populated_bins, x_f),
        #     bounds=((-15.0, 15.0),)
        # ).x
        #
        # # min_scale = optimize.fsolve(
        # #     lambda _scale: rmsd(_scale, y, r, y_inds, sample_grid, x_f),
        # #     0.0
        # # )
        # finish_solve = time.time()
        # print(f"\t\t\tSolve NEW SHGO 20: {finish_solve - begin_solve} with scale: {min_scale}")
        #
        # ####################### NEW 100 SHGO #########################
        #
        # # Get the resolution bins
        # sample_grid = np.linspace(np.min(r), np.max(r), 100)
        #
        # # Get the array that maps x values to bins
        # x_inds = np.digitize(reference_resolution_array, sample_grid)
        #
        # # Get the bin averages
        # populated_bins, counts = np.unique(x_inds, return_counts=True)
        # x_f = np.array([np.mean(x[x_inds == rb]) for rb in populated_bins[1:-2]])
        #
        # y_inds = np.digitize(dtag_resolution_array, sample_grid)
        #
        # finish_preprocess = time.time()
        # print(f"\t\t\tPreprocess: {finish_preprocess - begin_preprocess}")
        #
        # # Optimise the scale factor
        #
        # begin_solve = time.time()
        # # y_inds_unique = np.unique(y_inds)
        # min_scale = optimize.shgo(
        #     lambda _scale: get_rmsd(_scale, y, r, y_inds, populated_bins, x_f),
        #     bounds=((-15.0, 15.0),)
        # ).x
        #
        # # min_scale = optimize.fsolve(
        # #     lambda _scale: rmsd(_scale, y, r, y_inds, sample_grid, x_f),
        # #     0.0
        # # )
        # finish_solve = time.time()
        # print(f"\t\t\tSolve NEW SHGO 100: {finish_solve - begin_solve} with scale: {min_scale}")
        #
        # ########################## OLD #########################
        # begin_solve = time.time()
        #
        # sample_grid = np.linspace(min(r), max(r), 100)
        #
        # knn_x = neighbors.RadiusNeighborsRegressor(0.01)
        # knn_x.fit(r.reshape(-1, 1),
        #           x.reshape(-1, 1),
        #           )
        # x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)
        # # print(f"\t\t\txf OLD: {x_f}")
        #
        #
        # scales = []
        # rmsds = []
        #
        # knn_y = neighbors.RadiusNeighborsRegressor(0.01)
        # knn_y.fit(r.reshape(-1, 1),
        #           (y * np.exp(0.0 * r)).reshape(-1, 1),
        #           )
        #
        # y_neighbours = knn_y.radius_neighbors(sample_grid[:, np.newaxis])
        #
        # # Optimise the scale factor
        # for scale in np.linspace(-15, 15, 300):
        #     y_s = y * np.exp(scale * r)
        #
        #     y_f = np.array(
        #         [np.mean(y_s[y_neighbours[1][j]]) for j, val in enumerate(sample_grid[:, np.newaxis].flatten())])
        #
        #     _rmsd = np.sum(np.abs(x_f - y_f))
        #
        #     scales.append(scale)
        #     rmsds.append(_rmsd)
        #
        #
        # min_scale = scales[np.argmin(rmsds)]
        #
        # finish_solve = time.time()
        # print(f"\t\t\tSolve OLD: {finish_solve - begin_solve} with scale: {min_scale}")

        # Get the original reflections
        begin_dataset = time.time()
        original_reflections = dataset.reflections.reflections

        original_reflections_array = np.array(original_reflections,
                                              copy=True,
                                              )

        original_reflections_table = pd.DataFrame(original_reflections_array,
                                                  columns=reference_reflections.column_labels(),
                                                  )

        f_array = original_reflections_table[dataset.reflections.f]

        f_scaled_array = f_array * np.exp(min_scale * original_reflections.make_1_d2_array())

        original_reflections_table[dataset.reflections.f] = f_scaled_array

        # New reflections
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = original_reflections.spacegroup
        new_reflections.set_cell_for_all(original_reflections.cell)

        # Add dataset
        new_reflections.add_dataset("scaled")

        # Add columns
        for column in original_reflections.columns:
            new_reflections.add_column(column.label, column.type)

        # Update
        new_reflections.set_data(original_reflections_table.to_numpy())

        # Update resolution
        new_reflections.update_reso()


        # Create new dataset
        smoothed_dataset = XRayDataset(
            dataset.structure,
            Reflections(dataset.reflections.path,
                        dataset.reflections.f,
                        dataset.reflections.phi,
                        new_reflections),
            dataset.ligand_files
            # smoothing_factor=float(min_scale)
        )
        finish_dataset = time.time()
        print(f"\t\t\tMake dataset: {finish_dataset-begin_dataset}")

        finish_smooth_reflections = time.time()
        print(f"\t\tSmooth: {finish_smooth_reflections-begin_smooth_reflections}")

        return smoothed_dataset

    def real_space_smooth(self, dataset: DatasetInterface, grid_mask, exact_size):

        begin_smooth_reflections = time.time()

        # # Get common set of reflections
        begin_common = time.time()
        common_reflections_set = common_reflections({"reference" : self.reference_dataset, "dtag": dataset})
        finish_common = time.time()
        print(f"\t\t\tCommon: {finish_common-begin_common} with shape {common_reflections_set.shape}")

        # # Truncate
        begin_truncate = time.time()
        reference_reflections = truncate_reflections(
            self.reference_dataset.reflections.reflections, common_reflections_set)
        dtag_reflections = truncate_reflections(
            dataset.reflections.reflections, common_reflections_set)
        finish_truncate = time.time()
        print(f"\t\t\tTruncate: {finish_truncate-begin_truncate}")

        # Refference array
        # reference_reflections = truncated_reference.reflections.reflections
        begin_preprocess = time.time()
        reference_reflections_array = np.array(reference_reflections,
                                               copy=True,
                                               )
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                                   columns=reference_reflections.column_labels(),
                                                   )
        reference_f_array = reference_reflections_table[self.reference_dataset.reflections.f].to_numpy()


        # Dtag array
        # dtag_reflections = truncated_dataset.reflections.reflections
        dtag_reflections_array = np.array(dtag_reflections,
                                          copy=True,
                                          )
        dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                              columns=dtag_reflections.column_labels(),
                                              )
        dtag_f_array = dtag_reflections_table[dataset.reflections.f].to_numpy()
        print(f"\t\t\tReference f array size: {reference_f_array.shape} and dtag f array size: {dtag_f_array.shape}")

        # Resolution array
        # reference_resolution_array = reference_reflections.make_1_d2_array()
        # dtag_resolution_array = dtag_reflections.make_1_d2_array()

        # Prepare optimisation
        # x = reference_f_array
        # y = dtag_f_array

        # r = dtag_resolution_array

        # Get the resolution bins
        # sample_grid = np.linspace(np.min(r), np.max(r), 20)

        # Get the array that maps x values to bins
        # x_inds = np.digitize(reference_resolution_array, sample_grid)

        # Get the bin averages
        # populated_bins, counts = np.unique(x_inds, return_counts=True)
        # x_f = np.array([np.mean(x[x_inds == rb]) for rb in populated_bins[1:-2]])
        # print(f"\t\t\tsample NEW: {x_f}")
        # print(f"\t\t\txf NEW: {x_f}")

        # y_inds = np.digitize(dtag_resolution_array, sample_grid)

        finish_preprocess = time.time()
        print(f"\t\t\tPreprocess: {finish_preprocess - begin_preprocess}")

        # Optimise the scale factor

        reference_grid = self.reference_dataset.reflections.transform_f_phi_to_map(exact_size=exact_size)
        reference_array = np.array(reference_grid, copy=False)
        reference_values = reference_array[grid_mask]

        original_reflections = dataset.reflections.reflections
        original_reflections_array = np.array(original_reflections,
                                              copy=True,
                                              )
        original_reflections_table = pd.DataFrame(original_reflections_array,
                                                  columns=reference_reflections.column_labels(),
                                                  )
        f_array = original_reflections_table[dataset.reflections.f]
        r = original_reflections.make_1_d2_array()

        begin_solve = time.time()
        # y_inds_unique = np.unique(y_inds)
        min_scale = optimize.shgo(
            lambda _scale: get_rmsd_real_space(
                _scale,
                reference_values,
                f_array,
                r,
                grid_mask,
                dataset.reflections,
                exact_size,
            ),
            # 0.0,
            bounds=((-15.0, 15.0),),
            # tol=0.1
        ).x

        # min_scale = optimize.fsolve(
        #     lambda _scale: rmsd(_scale, y, r, y_inds, sample_grid, x_f),
        #     0.0
        # )
        finish_solve = time.time()
        print(f"\t\t\tSolve NEW BOUNDED 20: {finish_solve - begin_solve} with scale: {min_scale}")

        # Get the original reflections
        begin_dataset = time.time()
        original_reflections = dataset.reflections.reflections

        original_reflections_array = np.array(original_reflections,
                                              copy=True,
                                              )

        original_reflections_table = pd.DataFrame(original_reflections_array,
                                                  columns=reference_reflections.column_labels(),
                                                  )

        f_array = original_reflections_table[dataset.reflections.f]

        f_scaled_array = f_array * np.exp(min_scale * original_reflections.make_1_d2_array())

        original_reflections_table[dataset.reflections.f] = f_scaled_array

        # New reflections
        new_reflections = gemmi.Mtz(with_base=False)

        # Set dataset properties
        new_reflections.spacegroup = original_reflections.spacegroup
        new_reflections.set_cell_for_all(original_reflections.cell)

        # Add dataset
        new_reflections.add_dataset("scaled")

        # Add columns
        for column in original_reflections.columns:
            new_reflections.add_column(column.label, column.type)

        # Update
        new_reflections.set_data(original_reflections_table.to_numpy())

        # Update resolution
        new_reflections.update_reso()


        # Create new dataset
        smoothed_dataset = XRayDataset(
            dataset.structure,
            Reflections(dataset.reflections.path,
                        dataset.reflections.f,
                        dataset.reflections.phi,
                        new_reflections),
            dataset.ligand_files
            # smoothing_factor=float(min_scale)
        )
        finish_dataset = time.time()
        print(f"\t\t\tMake dataset: {finish_dataset-begin_dataset}")

        finish_smooth_reflections = time.time()
        print(f"\t\tSmooth: {finish_smooth_reflections-begin_smooth_reflections}")

        return smoothed_dataset
