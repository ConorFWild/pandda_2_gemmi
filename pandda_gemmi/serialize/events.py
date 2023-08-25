import numpy as np

import yaml

from pandda_gemmi.interfaces import *

from pandda_gemmi.event_model.event import Event


def serialize_events(
        selected_model_events: Dict[int, EventInterface],
        path
):
    dic = {

        event_idx: {
            "Score": selected_model_events[event_idx].score,
            "BDC": selected_model_events[event_idx].bdc,
            "Position Array": selected_model_events[event_idx].pos_array.tolist(),
            "Point Array": selected_model_events[event_idx].point_array.tolist()
        }
        for event_idx
        in sorted(selected_model_events)
    }

    with open(path, 'w') as f:
        yaml.dump(dic, f, sort_keys=False)


def unserialize_events(path):
    with open(path, 'r') as f:
        dic = yaml.safe_load(f)

    return {
        event_idx: Event(
            pos_array=np.array(event["Position Array"]),
            point_array=np.array(event["Point Array"]),
            score=event["Score"],
            bdc=event["BDC"]
        )
        for event_idx, event
        in dic.items()
    }
