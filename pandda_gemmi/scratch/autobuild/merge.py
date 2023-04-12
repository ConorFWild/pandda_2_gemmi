from .. import constants
from ..interfaces import *


def merge_build(dataset, selected_build_path, path):
    ...


def merge_autobuilds(datasets, autobuilds, fs: PanDDAFSInterface, build_selection_method):
    all_dtags = list(set([event_id[0] for event_id in autobuilds]))

    for dtag in all_dtags:
        dataset = datasets[dtag]
        dtag_events = [event_id for event_id in autobuilds if event_id[0] == dtag]
        selected_build_path = build_selection_method(dtag_events)
        model_building_dir = fs.output.processed_datasets[dtag] / constants.PANDDA_MODELLED_STRUCTURES_DIR
        merge_build(dataset, selected_build_path, model_building_dir / constants.PANDDA_EVENT_MODEL.format(dtag=dtag))


class MergeHighestRSCC:
    ...
