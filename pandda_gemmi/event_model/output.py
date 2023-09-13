import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import gemmi

from ..interfaces import *
from pandda_gemmi import constants
from ..dmaps import SparseDMap, save_dmap


def output_models(fs, characterization_sets, selected_model_num):
    ...


def output_events(fs, model_events):
    ...


def output_maps(
        dtag,
        fs,
        selected_events: Dict[Tuple[str, int], EventInterface],
        dtag_array,
        selected_mean,
        selected_z,
        reference_frame,
        res,
        all_events,
        model_means,
        all_maps=True
):
    zmap_grid = reference_frame.unmask(SparseDMap(selected_z))
    save_dmap(zmap_grid, fs.output.processed_datasets[dtag] / constants.PANDDA_Z_MAP_FILE.format(dtag=dtag))

    mean_grid = reference_frame.unmask(SparseDMap(selected_mean))
    save_dmap(mean_grid, fs.output.processed_datasets[dtag] / constants.PANDDA_MEAN_MAP_FILE.format(dtag=dtag))

    xmap_grid = reference_frame.unmask(SparseDMap(dtag_array))
    save_dmap(xmap_grid, fs.output.processed_datasets[dtag] / "xmap.ccp4")

    # mean_inner_vals = selected_mean[reference_frame.mask.indicies_sparse_inner_atomic]
    # med = np.median(mean_inner_vals)
    # from scipy import optimize

    for event_id, event in selected_events.items():
        # centroid = np.mean(event.pos_array, axis=0)
        # dist = np.linalg.norm(centroid - [6.0, -4.0, 25.0])
        # if dist < 5.0:
        #     print(f"##### {event_id} #####")
        #     print(centroid)
        #     xmap_grid = reference_frame.unmask(SparseDMap(dtag_array))
        #     xmap_array = np.array(xmap_grid, copy=False)
        #     mean_array = np.array(mean_grid, copy=False)
        #     event_indicies = tuple(
        #         [
        #             event.point_array[:, 0].flatten(),
        #             event.point_array[:, 1].flatten(),
        #             event.point_array[:, 2].flatten(),
        #         ]
        #     )
        #
        #     xmap_vals = xmap_array[event_indicies]
        #     mean_map_vals = mean_array[event_indicies]
        #     fig, axs = plt.subplots(21, 1, figsize=(6, 21*2))
        #
        #
        #
        #     res = optimize.minimize(
        #         lambda _bdc: np.abs(
        #             np.median(
        #                 (xmap_vals - (_bdc * mean_map_vals)) / (1 - _bdc)
        #             ) - med
        #     ),
        #         0.5,
        #         bounds=((0.0, 0.95),),
        #         # tol=0.1
        #     )
        #     print(res.x)
        #
        #     for j, bdc in enumerate([x for x in np.linspace(0.0,0.95, 20)] + [res.x,]):
        #         bdc = round(float(bdc), 2)
        #         event_map_vals = (xmap_vals - (bdc * mean_map_vals)) / (1 - bdc)
        #         table = pd.DataFrame([
        #             {"val": float(event_map_val),
        #              "type": "event"}
        #             for event_map_val
        #             in event_map_vals
        #         ] + [
        #             {"val": float(mean_inner_val),
        #              "type": "inner"}
        #             for mean_inner_val
        #             in mean_inner_vals
        #         ])
        #
        #         # sns.histplot(data=table, x="val", hue="type", ax=axs[j], kde=True, stat="density")
        #         sns.kdeplot(data=table, x="val", hue="type", ax=axs[j], common_norm=False)#, kde=True, stat="density")
        #         # sns.kdeplot(data=table, x="val", hue="type", ax=axs[j])#, kde=True, stat="density")
        #
        #     fig.savefig(
        #         Path(fs.output.processed_datasets[event_id[0]]) / f"{event_id[1]}_dist.png"
        #     )
        #     plt.clf()

            # event_array = (dtag_array - (bdc * selected_mean)) / (1 - bdc)
                #
                # event_grid = reference_frame.unmask(SparseDMap(event_array))


                # save_dmap(
                #     event_grid,
                #     # event_grid_smoothed,
                #     Path(fs.output.processed_datasets[event_id[0]]) / constants.PANDDA_EVENT_MAP_FILE.format(
                #         dtag=event_id[0],
                #         event_idx=event_id[1],
                #         bdc=round(1 - bdc, 2)
                #     ),
                #     np.mean(event.pos_array, axis=0),
                #     reference_frame
                # )

        event_array = (dtag_array - (event.bdc * selected_mean)) / (1 - event.bdc)
        event_grid = reference_frame.unmask(SparseDMap(event_array))

        # event_grid_recip = gemmi.transform_map_to_f_phi(event_grid,)
        # data = event_grid_recip.prepare_asu_data(dmin=res)
        # event_grid_smoothed = data.transform_f_phi_to_map(sample_rate=4.0)

        save_dmap(
            event_grid,
            # event_grid_smoothed,
            Path(fs.output.processed_datasets[event_id[0]]) / constants.PANDDA_EVENT_MAP_FILE.format(
                dtag=event_id[0],
                event_idx=event_id[1],
                bdc=round(1-event.bdc, 2)
            ),
            np.mean(event.pos_array, axis=0),
            reference_frame
        )

    if all_maps:
        all_maps_dir = Path(fs.output.processed_datasets[dtag]) / "all_maps"
        os.mkdir(all_maps_dir)
        for model_number, model_events in all_events:
            model_maps_dir = all_maps_dir / str(model_number)
            model_mean = model_means[model_number]
            os.mkdir(model_maps_dir)
            for event_id, event in model_events.items():
                event_array = (dtag_array - (event.bdc * model_mean)) / (1 - event.bdc)
                event_grid = reference_frame.unmask(SparseDMap(event_array))

                save_dmap(
                    event_grid,
                    # event_grid_smoothed,
                    Path(fs.output.processed_datasets[event_id[0]]) / constants.PANDDA_EVENT_MAP_FILE.format(
                        dtag=event_id[0],
                        event_idx=event_id[1],
                        bdc=round(1 - event.bdc, 2)
                    ),
                    np.mean(event.pos_array, axis=0),
                    reference_frame
                )

