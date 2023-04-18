import numpy as np
import gemmi

from ..interfaces import *
from .. import constants
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
        res
):
    zmap_grid = reference_frame.unmask(SparseDMap(selected_z))
    save_dmap(zmap_grid, fs.output.processed_datasets[dtag] / constants.PANDDA_Z_MAP_FILE.format(dtag=dtag))

    mean_grid = reference_frame.unmask(SparseDMap(selected_mean))
    save_dmap(mean_grid, fs.output.processed_datasets[dtag] / constants.PANDDA_MEAN_MAP_FILE.format(dtag=dtag))

    for event_id, event in selected_events.items():
        centroid = np.mean(event.pos_array, axis=0)
        dist = np.linalg.norm(centroid - [6.0, -4.0, 25.0])
        if dist < 5.0:
            for bdc in np.linspace(0.0,0.95, 20):
                bdc = round(float(bdc), 2)
                event_array = (dtag_array - (bdc * selected_mean)) / (1 - bdc)
                event_grid = reference_frame.unmask(SparseDMap(event_array))
                save_dmap(
                    event_grid,
                    # event_grid_smoothed,
                    Path(fs.output.processed_datasets[event_id[0]]) / constants.PANDDA_EVENT_MAP_FILE.format(
                        dtag=event_id[0],
                        event_idx=event_id[1],
                        bdc=round(1 - bdc, 2)
                    ),
                    np.mean(event.pos_array, axis=0),
                    reference_frame
                )

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

