import numpy as np

from ..interfaces import *

def evaluate_model(
        dtag,
        datasets,
        characterization_set,
                dmaps,
                statistical_model,
                cluster_density,
                pre_score_filters,
                scoring,
                post_score_filters):

    # Get the relevant dmaps
    dtag_array = np.array([_dtag for _dtag in datasets])

    # Get the dataset dmap
    dtag_index = np.argwhere(dtag_array == dtag)
    dataset_dmap_array = dmaps[dtag_index,:]

    # Get the characterization set dmaps
    characterization_set_mask_list = []
    for _dtag in datasets:
        if _dtag in characterization_set:
            characterization_set_mask_list.append(True)
        else:
            characterization_set_mask_list.append(False)
    characterization_set_mask = np.array([characterization_set_mask_list])
    characterization_set_dmaps = dmaps[characterization_set_mask, :]

    # Get the statical maps
    statistical_maps = statistical_model()

    # Initial
    events = cluster_density()

    #
    for filter in pre_score_filters:
        events = filter(events)

    #
    events = scoring(events)

    #
    for filter in post_score_filters:
        events = filter(events)

    return events
