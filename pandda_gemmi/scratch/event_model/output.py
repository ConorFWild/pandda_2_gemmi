import numpy as np
import gemmi

from ..interfaces import *
from .. import constants
from ..dmaps import SparseDMap

def output_models(fs, characterization_sets, selected_model_num):
    ...

def output_events(fs, model_events):
    ...


def save_dmap(dmap, path, centroid=None, reference_frame: DFrameInterface=None, radius=15.0):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = dmap
    ccp4.update_ccp4_header(2, True)
    if reference_frame:
        box = gemmi.FractionalBox()
        cart_max = centroid + radius
        cart_min = centroid - radius
        unit_cell = reference_frame.get_grid().unit_cell
        frac_max = unit_cell.fractionalize(gemmi.Position(*cart_max))
        frac_min = unit_cell.fractionalize(gemmi.Position(*cart_min))
        box.extend(frac_max)
        box.extend(frac_min)
        ccp4.set_extent(box)
    ccp4.write_ccp4_map(str(path))

def output_maps(
        dtag,
        fs,
        selected_events: Dict[Tuple[str, int], EventInterface],
        dtag_array,
        selected_mean,
selected_z,
        reference_frame
):
    zmap_grid = reference_frame.unmask(SparseDMap(selected_z))
    save_dmap(zmap_grid, fs.output.processed_datasets[dtag] / constants.PANDDA_Z_MAP_FILE.format(dtag=dtag))

    for event_id, event in selected_events.items():
        event_array = (dtag_array - (event.bdc * selected_mean)) / (1 - event.bdc)
        event_grid = reference_frame.unmask(SparseDMap(event_array))

        save_dmap(
            event_grid,
            Path(fs.output.processed_datasets[event_id[0]]) / constants.PANDDA_EVENT_MAP_FILE.format(
                dtag=event_id[0],
                event_idx=event_id[1],
                bdc=round(event.bdc, 2)
            ),
            np.mean(event.pos_array, axis=0),
            reference_frame
        )