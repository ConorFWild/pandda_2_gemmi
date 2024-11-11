import numpy as np

import yaml

from pandda_gemmi.interfaces import *


def processed_dataset(
        comparator_datasets: Dict[str, DatasetInterface],
        processing_res: float,
        characterization_sets: Dict[int, Dict[str, DatasetInterface]],
        model_scores: Dict[int, float],
        models_to_process: List[int],
        processed_models: Dict[int, Tuple[Dict[int, EventInterface], Any, Any]],
        selected_model_num: int,
        selected_model_events: Dict[ Tuple[str, int], EventInterface],
        reference_frame: DFrameInterface,
        model_metas,
        path
):
    dic = {}
    dic["Summary"] = {
        "Processing Resolution": round(processing_res, 2),
        "Comparator Datasets": sorted([x for x in comparator_datasets]),
        "Selected Model": selected_model_num,
        "Selected Model Events": sorted([x[1] for x in selected_model_events])
    }

    dic["Models"] = {}

    for model_num in sorted(characterization_sets):

        if model_num in models_to_process:
            is_processed = True
            model_events, z, mean, std, meta = processed_models[model_num]
            if model_events is None:
                model_events = {}

        else:
            is_processed = False
            model_events = {}
            meta = {
            'Number of Initial Events': 0,
            'Number of Size Filtered Events': 0,
            'Number of Score Filtered Events': 0
            }


        # inner_mask_zmap = z[reference_frame.mask.indicies_sparse_inner_atomic]
        # percentage_z_2: float = float(np.sum(np.abs(inner_mask_zmap) > 2)) / inner_mask_zmap.size

        dic["Models"][model_num] = {
            "Processed?": is_processed,
            "Characterization Datasets": [x for x in characterization_sets[model_num]],
            "Model Score": round(model_scores[model_num], 2),
            'Number of Initial Events': meta['Number of Initial Events'],
            'Number of Size Filtered Events': meta['Number of Size Filtered Events'],
            'Number of Score Filtered Events':meta['Number of Score Filtered Events'],
            "Events": {
                event_idx: {
                    "Score": model_events[event_idx].score,
                    "BDC": model_events[event_idx].bdc,
                    "Size": model_events[event_idx].size,
                    "Centroid": [float(x) for x in np.mean(model_events[event_idx].pos_array, axis=0).flatten()],
                    "score": model_events[event_idx].build.score,
                    "Local Strength": float(model_events[event_idx].local_strength),
                    "Build Score": model_events[event_idx].build.build_score,
                    "Noise": model_events[event_idx].build.noise,
                    "Signal": model_events[event_idx].build.signal,
                    'Num. Contacts': model_events[event_idx].build.num_contacts,
                    "Num. Points": float(model_events[event_idx].build.num_points),
                    'Optimal Contour': float(model_events[event_idx].build.optimal_contour),
                    "RSCC": float(model_events[event_idx].build.rscc),
                    "Ligand Centroid": [float(x) for x in model_events[event_idx].build.centroid],
                    "Build Path": str(model_events[event_idx].build.build_path),
                    "Ligand Key": str(model_events[event_idx].build.ligand_key)
                }
                for event_idx
                in sorted(model_events)
            }
        }

    with open(path, 'w') as f:
        yaml.dump(dic, f, sort_keys=False)

    ...
