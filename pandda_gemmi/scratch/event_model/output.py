import gemmi

from ..interfaces import *
from .. import constants
from ..dmaps import SparseDMap

def output_models(fs, characterization_sets, selected_model_num):
    ...

def output_events(fs, model_events):
    ...


def save_dmap(dmap, path):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = dmap
    ccp4.update_ccp4_header(2, True)
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
            )
        )