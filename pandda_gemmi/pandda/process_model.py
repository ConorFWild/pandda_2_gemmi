import time

# try:
#     from sklearnex import patch_sklearn
#
#     patch_sklearn()
# except ImportError:
#     print('No sklearn-express available!')

import numpy as np
import gemmi

from pandda_gemmi.interfaces import *

from pandda_gemmi.dmaps import (
    SparseDMap,
)

from pandda_gemmi.event_model.outlier import PointwiseMAD
from pandda_gemmi.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.event_model.filter import (
    FilterSize,
    FilterScore,
)
from pandda_gemmi.event_model.get_bdc import get_bdc

from pandda_gemmi.autobuild.inbuilt import get_conformers

from pandda_gemmi.cnn import set_structure_mean


class ProcessModel:
    def __init__(self,
                 minimum_z_cluster_size=5.0,
                 minimum_event_score=0.15,
                 local_highest_score_radius=8.0,
                 use_ligand_data=True,
                 debug=False
                 ):
        self.minimum_z_cluster_size = minimum_z_cluster_size
        self.minimum_event_score = minimum_event_score
        self.local_highest_score_radius = local_highest_score_radius
        self.use_ligand_data = use_ligand_data
        self.debug = debug

    def __call__(self,  # *args, **kwargs
                 ligand_files,
                 homogenized_dataset_dmap_array,
                 dataset_dmap_array,
                 characterization_set_dmaps_array,
                 reference_frame,
                 model_map,
                 score,
                 fs,
                 model_number,
                 dtag
                 ):
        # Get the statical maps
        mean, std, z = PointwiseMAD()(
            homogenized_dataset_dmap_array,
            characterization_set_dmaps_array
        )

        mean_grid = reference_frame.unmask(SparseDMap(mean))
        z_grid = reference_frame.unmask(SparseDMap((z - np.mean(z)) / np.std(z)))
        if self.debug:
            print(
                f'model {model_number} xmap stats: min {np.min(homogenized_dataset_dmap_array)} max {np.max(homogenized_dataset_dmap_array)} mean {np.mean(homogenized_dataset_dmap_array)}')
            print(f'model {model_number} mean stats: min {np.min(mean)} max {np.max(mean)} mean {np.mean(mean)}')
            print(f'model {model_number} std stats: min {np.min(std)} max {np.max(std)} mean {np.mean(std)}')
            print(f'model {model_number} z stats: min {np.min(z)} max {np.max(z)} mean {np.mean(z)}')

        # Get the median
        protein_density_median = np.median(mean[reference_frame.mask.indicies_sparse_inner_atomic])

        # unsparsify input maps
        xmap_grid = reference_frame.unmask(SparseDMap(homogenized_dataset_dmap_array))
        raw_xmap_grid = gemmi.FloatGrid(*dataset_dmap_array.shape)
        raw_xmap_grid.set_unit_cell(z_grid.unit_cell)
        raw_xmap_grid_array = np.array(raw_xmap_grid, copy=False)
        raw_xmap_grid_array[:, :, :] = dataset_dmap_array[:, :, :]
        model_grid = reference_frame.unmask(SparseDMap(model_map))

        # Get the initial events from clustering the Z map
        events, cluster_metadata = ClusterDensityDBSCAN()(z, reference_frame)
        cutoff, high_z_all_points_mask, eps = cluster_metadata.values()
        num_initial_events = len(events)
        if self.debug:
            print(
                f'model {model_number}: Z map cutoff: {round(cutoff, 2)} results in {num_initial_events} events from {np.sum(high_z_all_points_mask)} high z points and eps {eps}')

        # Handle the edge case of zero events
        if len(events) == 0:
            return None, mean, z, std, {}

        # Filter the events prior to scoring them based on their size
        for filter in [
            FilterSize(reference_frame, min_size=self.minimum_z_cluster_size),
        ]:
            events = filter(events)
        num_size_filtered_events = len(events)

        if self.debug & (num_size_filtered_events > 0):
            size_range = (min([_event.pos_array.shape[0] for _event in events.values()]),
                          max([_event.pos_array.shape[0] for _event in events.values()]))

            print(
                f'model {model_number}: size filtering results in {num_size_filtered_events} with volume element {round(reference_frame.get_grid().unit_cell.volume / reference_frame.get_grid().point_count, 2)} and size range {size_range}')

        # Return None if there are no events after pre-scoring filters
        if len(events) == 0:
            return None, mean, z, std, {}

        # Score the events with some method such as the CNN
        time_begin_score_events = time.time()
        # events = score(ligand_files, events, xmap_grid, raw_xmap_grid, mean_grid, z_grid, model_grid,
        #                median, reference_frame, homogenized_dataset_dmap_array, mean
        #                )
        print(f'{self.use_ligand_data} {type(self.use_ligand_data)} {self.use_ligand_data == True}')
        if self.use_ligand_data:
            for lid, ligand_data in ligand_files.items():
                confs = get_conformers(ligand_data)
                for event_id, event in events.items():
                    conf = set_structure_mean(confs[0], event.centroid)
                    event_score, map_array, mol_array = score(
                        event,
                        conf,
                        z_grid,
                        raw_xmap_grid
                    )
                    if event_score > event.score:
                        event.score = event_score
                    _x, _y, _z, = event.centroid
                    # print(f'\t {model_number}_{event_id}_{lid}: ({_x}, {_y}, {_z}): {round(event_score, 5)}')

                    # dmaps = {
                    #     'zmap': map_array[0][0],
                    #     'xmap': map_array[0][1],
                    #     'mask': mol_array[0][0],
                    # }
                    # for name, dmap in dmaps.items():
                    #     grid = gemmi.FloatGrid(32, 32, 32)
                    #     uc = gemmi.UnitCell(16.0, 16.0, 16.0, 90.0, 90.0, 90.0)
                    #
                    #     # uc = gemmi.UnitCell(8.0, 8.0, 8.0, 90.0, 90.0, 90.0)
                    #     grid.set_unit_cell(uc)
                    #
                    #     grid_array = np.array(grid, copy=False)
                    #     grid_array[:, :, :] = dmap[:, :, :]
                    #     ccp4 = gemmi.Ccp4Map()
                    #     ccp4.grid = grid
                    #     ccp4.update_ccp4_header()
                    #     ccp4.write_ccp4_map(
                    #         str(fs.output.processed_datasets[dtag] / f'{model_number}_{event_id}_{lid}_{name}.ccp4'))

                time_finish_score_events = time.time()
        else:
            for event_id, event in events.items():
                event_score, map_array, mol_array = score(
                    event,
                    None,
                    z_grid,
                    raw_xmap_grid
                )
                event.score = event_score
                _x, _y, _z, = event.centroid
                event.bdc = get_bdc(event, xmap_grid, mean_grid, protein_density_median)

        # Filter the events after scoring based on keeping only the locally highest scoring event
        num_events = len(events)
        score_range = (round(min([_event.score for _event in events.values()]), 2),
                       round(max([_event.score for _event in events.values()]), 2))
        for filter in [
            FilterScore(self.minimum_event_score),  # Filter events based on their score
            # FilterLocallyHighestLargest(self.local_highest_score_radius),  # Filter events that are close to other,
            #                                                                # better scoring events
            # FilterLocallyHighestScoring(self.local_highest_score_radius)
        ]:
            events = filter(events)
        # TODO: Replace with logger printing
        num_score_filtered_events = len(events)

        if self.debug:
            print(
                f'model {model_number}: score filtering results in {num_score_filtered_events} with cutoff {self.minimum_event_score} and score range {score_range}')

        # Return None if there are no events after post-scoring filters
        if len(events) == 0:
            return None, mean, z, std, {}

        # Renumber the events
        events = {j + 1: event for j, event in enumerate(events.values())}

        # print(f'z map stats: {np.min(z)} {np.max(z)} {np.median(z)} {np.sum(np.isnan(z))}')

        meta = {
            'Number of Initial Events': num_initial_events,
            'Number of Size Filtered Events': num_size_filtered_events,
            'Number of Score Filtered Events': num_score_filtered_events
        }

        return events, mean, z, std, meta
