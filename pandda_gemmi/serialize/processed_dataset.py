import numpy as np

import yaml

from pandda_gemmi.interfaces import *


def processed_dataset(
        comparator_datasets: Dict[str, DatasetInterface],
        processing_res: float,
        characterization_sets: Dict[int, Dict[str, DatasetInterface]],
        models_to_process: List[int],
        processed_models: Dict[int, Tuple[Dict[Tuple[str, int], EventInterface], Any, Any]],
        selected_model_num: int,
        selected_model_events: Dict[ int, EventInterface],
        reference_frame: DFrameInterface,
        path
):
    dic = {}
    dic["Summary"] = {
        "Processing Resolution": round(processing_res, 2),
        "Comparator Datasets": sorted([x for x in comparator_datasets]),
        "Selected Model": selected_model_num,
        "Selected Model Events": sorted([x for x in selected_model_events])
    }

    dic["Models"] = {}

    for model_num in sorted(characterization_sets):

        model_events, z, mean = processed_models[model_num]

        if model_num in models_to_process:
            is_processed = True
        else:
            is_processed = False

        inner_mask_zmap = z[reference_frame.mask.indicies_sparse_inner_atomic]
        percentage_z_2: float = float(np.sum(np.abs(inner_mask_zmap) > 2)) / inner_mask_zmap.size

        dic["Models"][model_num] = {
            "Processed?": is_processed,
            "Characterization Datasets": [x for x in characterization_sets[model_num]],
            "Percentage Z > 2": round(percentage_z_2, 2),
            "Events": {
                event_idx: {
                    "Score": model_events[event_idx].score,
                    "BDC": model_events[event_idx].bdc,
                }
                for event_idx
                in sorted(model_events)
            }
        }

    with open(path, 'w') as f:
        yaml.dump(dic, f)

    ...
