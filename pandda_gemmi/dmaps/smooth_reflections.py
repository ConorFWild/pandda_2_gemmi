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


    _rmsd = np.sum(np.abs(reference_values.flatten() - masked_new_grid_values.flatten()))
    return _rmsd

class SmoothReflections:
    def __init__(self, dataset: DatasetInterface):
        self.reference_dataset = dataset

    def __call__(self, dataset: DatasetInterface):

        # # Get common set of reflections
        common_reflections_set = common_reflections(
            {"reference" : self.reference_dataset,
             "dtag": dataset},
        )

        # # Truncate
        reference_reflections, ref_mask_non_zero = truncate_reflections(
            self.reference_dataset.reflections.reflections,
            common_reflections_set,
        )
        dtag_reflections, dtag_mask_non_zero = truncate_reflections(
            dataset.reflections.reflections,
            common_reflections_set,
        )

        # Refference array
        reference_reflections_array = np.array(reference_reflections,
                                               copy=True,
                                               )
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                                   columns=reference_reflections.column_labels(),
                                                   )
        reference_f_array = reference_reflections_table[self.reference_dataset.reflections.f].to_numpy()


        # Dtag array
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

        # Get the resolution bins
        sample_grid = np.linspace(np.min(r), np.max(r), 20)

        # Get the array that maps x values to bins
        x_inds = np.digitize(reference_resolution_array, sample_grid)

        # Get the bin averages
        populated_bins, counts = np.unique(x_inds, return_counts=True)
        x_f = np.array([np.mean(x[x_inds == rb]) for rb in populated_bins[1:-2]])

        y_inds = np.digitize(dtag_resolution_array, sample_grid)


        # Optimise the scale factor
        try:
            min_scale = optimize.minimize(
                lambda _scale: get_rmsd(_scale, y, r, y_inds, populated_bins, x_f),
                0.0,
                bounds=((-15.0, 15.0),),
                tol=0.1
            ).x
        except Exception as e:
            print("######## Ref hkl")
            data_array = np.array(reference_reflections, copy=False)
            data_hkl = data_array[:, 0:3].astype(int)
            print(data_hkl)
            print(data_hkl.shape)
            print("######## dtag hkl")
            data_array = np.array(dtag_reflections, copy=False)
            data_hkl = data_array[:, 0:3].astype(int)
            print(data_hkl)
            print(data_hkl.shape)
            print("######## Ref mask non zero")
            print(ref_mask_non_zero)
            print("######## dtag mask non zero")
            print(dtag_mask_non_zero)
            print("######## Reference f array / x")
            print(reference_f_array)
            print(reference_f_array.size)
            print("######## dtag f array / y")
            print(dtag_f_array)
            print(dtag_f_array.size)
            print("######## reference_resolution_array / r")
            print(reference_resolution_array)
            print(reference_resolution_array.size)
            print("######## Dtag resolution array")
            print(dtag_resolution_array)
            print(dtag_resolution_array.size)
            print("######## y inds")
            print(y_inds)
            print(y_inds.size)
            print("######## x_f")
            print(x_f)
            print(x_f.size)
            raise Exception


        original_reflections = dataset.reflections.reflections

        original_reflections_array = np.array(original_reflections,
                                              copy=True,
                                              )

        original_reflections_table = pd.DataFrame(original_reflections_array,
                                                  columns=dtag_reflections.column_labels(),
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
        )

        return smoothed_dataset

    def real_space_smooth(self, dataset: DatasetInterface, grid_mask, exact_size):


        # # Get common set of reflections
        common_reflections_set = common_reflections({"reference" : self.reference_dataset, "dtag": dataset})

        # # Truncate
        reference_reflections = truncate_reflections(
            self.reference_dataset.reflections.reflections, common_reflections_set)
        dtag_reflections = truncate_reflections(
            dataset.reflections.reflections, common_reflections_set)

        # Refference array
        reference_reflections_array = np.array(reference_reflections,
                                               copy=True,
                                               )
        reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                                   columns=reference_reflections.column_labels(),
                                                   )
        reference_f_array = reference_reflections_table[self.reference_dataset.reflections.f].to_numpy()


        # Dtag array
        dtag_reflections_array = np.array(dtag_reflections,
                                          copy=True,
                                          )
        dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                              columns=dtag_reflections.column_labels(),
                                              )
        dtag_f_array = dtag_reflections_table[dataset.reflections.f].to_numpy()

        # Resolution array

        # Prepare optimisation

        # Get the resolution bins

        # Get the array that maps x values to bins

        # Get the bin averages

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
            bounds=((-15.0, 15.0),),
            sampling_method='sobol'
        ).x


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
        )

        return smoothed_dataset
