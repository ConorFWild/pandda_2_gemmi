import time

from ..interfaces import *
from ..dataset import Reflections, XRayDataset

import gemmi
import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rfn

dt = np.dtype([('h', 'i4'), ('k', 'i4'),('l', 'i4'),])
# def truncate_reflections(reflections, index=None):
#     new_reflections = gemmi.Mtz(with_base=False)
#
#     # Set dataset properties
#     new_reflections.spacegroup = reflections.spacegroup
#     new_reflections.set_cell_for_all(reflections.cell)
#
#     # Add dataset
#     new_reflections.add_dataset("truncated")
#
#     # Add columns
#     for column in reflections.columns:
#         new_reflections.add_column(column.label, column.type)
#
#     # Get data
#     data_array = np.array(reflections, copy=True)
#     data = pd.DataFrame(data_array,
#                         columns=reflections.column_labels(),
#                         )
#     data.set_index(["H", "K", "L"], inplace=True)
#     # print(data)
#     # print(self.reflections.make_miller_array().shape)
#
#     # Truncate by index
#     data_indexed = data.loc[index]
#
#     # To numpy
#     data_dropped_array = data_indexed.to_numpy()
#
#     # new data
#     new_data = np.hstack([data_indexed.index.to_frame().to_numpy(),
#                           data_dropped_array,
#                           ]
#                          )
#     # print(new_data)
#
#     # Update
#     new_reflections.set_data(new_data)
#
#     # Update resolution
#     new_reflections.update_reso()
#     # print(new_reflections.make_miller_array().shape)
#
#     return new_reflections

# def truncate_reflections(reflections, index=None):
#     new_reflections = gemmi.Mtz(with_base=False)
#
#     # Set dataset properties
#     new_reflections.spacegroup = reflections.spacegroup
#     new_reflections.set_cell_for_all(reflections.cell)
#
#     # Add dataset
#     new_reflections.add_dataset("truncated")
#
#     # Add columns
#     for column in reflections.columns:
#         new_reflections.add_column(column.label, column.type)
#
#     # Get data
#     data_array = np.array(reflections, copy=False)
#     # data = pd.DataFrame(data_array,
#     #                     columns=reflections.column_labels(),
#     #                     )
#     # data.set_index(["H", "K", "L"], inplace=True)
#     # hkl_array = data_array[:, 0:3]
#     # print(data)
#     # print(self.reflections.make_miller_array().shape)
#
#     # Truncate by index
#     # data_indexed = data.loc[index]
#
#     # To numpy
#     # data_dropped_array = data_indexed.to_numpy()
#
#     # new data
#     # new_data = np.hstack([data_indexed.index.to_frame().to_numpy(),
#     #                       data_dropped_array,
#     #                       ]
#     #                      )
#     structured_data_array = rfn.unstructured_to_structured(data_array[:, 0:3], dt)
#     new_data = data_array[np.in1d(structured_data_array, index)]
#     # print(new_data)
#
#     # Update
#     new_reflections.set_data(new_data)
#
#     # Update resolution
#     new_reflections.update_reso()
#     # print(new_reflections.make_miller_array().shape)
#
#     return new_reflections

def truncate_reflections(reflections, index=None):
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = reflections.spacegroup
    new_reflections.set_cell_for_all(reflections.cell)

    # Add dataset
    new_reflections.add_dataset("truncated")

    # Add columns
    for column in reflections.columns:
        new_reflections.add_column(column.label, column.type)

    # Get data
    data_array = np.array(reflections, copy=False)
    # data = pd.DataFrame(data_array,
    #                     columns=reflections.column_labels(),
    #                     )
    # data.set_index(["H", "K", "L"], inplace=True)
    # hkl_array = data_array[:, 0:3]
    # print(data)
    # print(self.reflections.make_miller_array().shape)

    # Truncate by index
    # data_indexed = data.loc[index]

    # To numpy
    # data_dropped_array = data_indexed.to_numpy()

    # new data
    # new_data = np.hstack([data_indexed.index.to_frame().to_numpy(),
    #                       data_dropped_array,
    #                       ]
    #                      )
    data_hkl = data_array[:, 0:3].astype(int)
    con_coords = np.vstack([data_hkl, index])
    size = np.max(con_coords, axis=0)-np.min(con_coords,axis=0)
    data_array_3d = np.zeros((size[0], size[1], size[2]), dtype=np.bool)
    data_array_3d[(index[:,0], index[:, 1], index[:, 2])] = True
    mask = data_array_3d[(data_hkl[:,0], data_hkl[:, 1], data_hkl[:, 2])]


    # structured_data_array = rfn.unstructured_to_structured(data_array[:, 0:3], dt)
    # new_data = data_array[np.in1d(structured_data_array, index)]
    new_data = data_array[mask]
    # print(new_data)

    # Update
    new_reflections.set_data(new_data)

    # Update resolution
    new_reflections.update_reso()
    # print(new_reflections.make_miller_array().shape)

    return new_reflections


def truncate_resolution(reflections, resolution: float):
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = reflections.spacegroup
    new_reflections.set_cell_for_all(reflections.cell)

    # Add dataset
    new_reflections.add_dataset("truncated")

    # Add columns
    for column in reflections.columns:
        new_reflections.add_column(column.label, column.type)

    # Get data
    data_array = np.array(reflections, copy=True)
    data = pd.DataFrame(data_array,
                        columns=reflections.column_labels(),
                        )
    data.set_index(["H", "K", "L"], inplace=True)

    # add resolutions
    data["res"] = reflections.make_d_array()

    # Truncate by resolution
    data_truncated = data[data["res"] >= resolution]

    # Rem,ove res colum
    data_dropped = data_truncated.drop("res", "columns")

    # To numpy
    data_dropped_array = data_dropped.to_numpy()

    # new data
    new_data = np.hstack([data_dropped.index.to_frame().to_numpy(),
                          data_dropped_array,
                          ]
                         )

    # Update
    new_reflections.set_data(new_data)

    # Update resolution
    new_reflections.update_reso()

    return new_reflections


# def common_reflections(reflections: Reflections,
#                        reference_ref: Reflections,
#                        ):
#     # Get own reflections
#     dtag_reflections = self.reflections.reflections
#     dtag_reflections_array = np.array(dtag_reflections, copy=True)
#     dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
#                                           columns=dtag_reflections.column_labels(),
#                                           )
#     dtag_reflections_table.set_index(["H", "K", "L"], inplace=True)
#     dtag_flattened_index = dtag_reflections_table[
#         ~dtag_reflections_table[structure_factors.f].isna()].index.to_flat_index()
#
#     # Get reference
#     reference_reflections = reference_ref.reflections
#     reference_reflections_array = np.array(reference_reflections, copy=True)
#     reference_reflections_table = pd.DataFrame(reference_reflections_array,
#                                                columns=reference_reflections.column_labels(),
#                                                )
#     reference_reflections_table.set_index(["H", "K", "L"], inplace=True)
#     reference_flattened_index = reference_reflections_table[
#         ~reference_reflections_table[structure_factors.f].isna()].index.to_flat_index()
#
#     running_index = dtag_flattened_index.intersection(reference_flattened_index)
#
#     return running_index.to_list()


# def common_reflections(datasets: Dict[str, DatasetInterface], tol=0.000001):
#     running_index: Optional[pd.Index] = None
#
#     for dtag in datasets:
#         dataset = datasets[dtag]
#         reflections = dataset.reflections.reflections
#         reflections_array = np.array(reflections, copy=True)
#         reflections_table = pd.DataFrame(reflections_array,
#                                          columns=reflections.column_labels(),
#                                          )
#         reflections_table.set_index(["H", "K", "L"], inplace=True)
#
#         is_na = reflections_table[dataset.reflections.f].isna()
#         is_zero = reflections_table[dataset.reflections.f].abs() < tol
#         mask = ~(is_na | is_zero)
#
#         flattened_index = reflections_table[mask].index.to_flat_index()
#         if running_index is None:
#             running_index = flattened_index
#         if running_index is not None:
#             running_index = running_index.intersection(flattened_index)
#
#     if running_index is not None:
#         return running_index.to_list()
#
#     else:
#         raise Exception(
#             "Somehow a running index has not been calculated. This should be impossible. Contact mantainer.")

# def common_reflections(datasets: Dict[str, DatasetInterface], tol=0.000001):
#     # running_index: Optional[pd.Index] = None
#
#     hkls = np.vstack(
#         [
#             np.array(datasets[dtag].reflections.reflections, copy=False)[:,0:3]
#             for dtag
#             in datasets
#         ]
#         )
#     structured_data_array = rfn.unstructured_to_structured(hkls, dt)
#
#     unique_rows, counts = np.unique(structured_data_array, return_counts=True)
#     common_rows = unique_rows[counts == len(datasets)]
#     return common_rows

def common_reflections(datasets: Dict[str, DatasetInterface], tol=0.000001):
    # running_index: Optional[pd.Index] = None

    hkl_arrays = [
            np.array(datasets[dtag].reflections.reflections, copy=False)[:,0:3].astype(int)
            for dtag
            in datasets
        ]

    hkls = np.vstack(
        hkl_arrays
        )
    size = np.max(hkls, axis=0)-np.min(hkls,axis=0)
    data_array_3d = np.zeros((size[0], size[1], size[2]), dtype=np.int)
    # data_array_3d = np.zeros((x for x in np.max(hkls, axis=0)-np.min(hkls,axis=0)), dtype=np.bool)
    for hkl_array in hkl_arrays:
        data_array_3d[(hkl_array[:, 0], hkl_array[:, 1], hkl_array[:, 2])] += 1


    # structured_data_array = rfn.unstructured_to_structured(hkls, dt)

    # unique_rows, counts = np.unique(structured_data_array, return_counts=True)
    # common_rows = unique_rows[counts == len(datasets)]
    common_rows = np.argwhere(data_array_3d == len(datasets))
    return common_rows

    # for dtag in datasets:
    #     dataset = datasets[dtag]
    #     reflections = dataset.reflections.reflections
    #     reflections_array = np.array(reflections, copy=True)
    #     reflections_table = pd.DataFrame(reflections_array,
    #                                      columns=reflections.column_labels(),
    #                                      )
    #     reflections_table.set_index(["H", "K", "L"], inplace=True)
    #
    #     is_na = reflections_table[dataset.reflections.f].isna()
    #     is_zero = reflections_table[dataset.reflections.f].abs() < tol
    #     mask = ~(is_na | is_zero)
    #
    #     flattened_index = reflections_table[mask].index.to_flat_index()
    #     if running_index is None:
    #         running_index = flattened_index
    #     if running_index is not None:
    #         running_index = running_index.intersection(flattened_index)
    #
    # if running_index is not None:
    #     return running_index.to_list()
    #
    # else:
    #     raise Exception(
    #         "Somehow a running index has not been calculated. This should be impossible. Contact mantainer.")


class TruncateReflections:
    def __init__(self,
                 datasets: Dict[str, DatasetInterface],
                 resolution: float,
                 ):
        # new_datasets_resolution = {}
        #
        # # Truncate by common resolution
        # for dtag in datasets:
        #     truncated_dataset = truncate_resolution(
        #         datasets[dtag],
        #         resolution,
        #     )
        #
        #     new_datasets_resolution[dtag] = truncated_dataset
        #
        # dataset_resolution_truncated = new_datasets_resolution

        # Get common set of reflections
        self.common_reflections_set = common_reflections(datasets)

        self.resolution = resolution

    def __call__(self, dataset: DatasetInterface):
        begin_truncate_reflections = time.time()
        new_reflections = Reflections(
            dataset.reflections.path,
            dataset.reflections.f,
            dataset.reflections.phi,
            # truncate_reflections(dataset.reflections.reflections, self.common_reflections_set)
            truncate_resolution(dataset.reflections.reflections, self.resolution)
        )

        new_dataset = XRayDataset(
            dataset.structure,
            new_reflections,
            dataset.ligand_files
        )
        finish_truncate_reflections = time.time()
        print(f"\t\tTruncate: {finish_truncate_reflections-begin_truncate_reflections}")

        return new_dataset

        # # truncate on reflections
        # new_datasets_reflections = {}
        # for dtag in dataset_resolution_truncated:
        #     reflections = dataset_resolution_truncated[dtag].reflections.reflections
        #     reflections_array = np.array(reflections)
        #
        #     print(f"Truncated reflections: {dtag}")
        #     truncated_dataset = dataset_resolution_truncated[dtag].truncate_reflections(common_reflections_set,
        #                                                                                 )
        #     truncated_dataset = truncate_reflections(reflections, index)
        #     reflections = truncated_dataset.reflections.reflections
        #     reflections_array = np.array(reflections)
        #
        #     new_datasets_reflections[dtag] = truncated_dataset
        #
        # return new_datasets_reflections
