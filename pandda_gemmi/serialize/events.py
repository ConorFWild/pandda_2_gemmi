import numpy as np

import yaml

from pandda_gemmi.interfaces import *

from pandda_gemmi.event_model.event import Event, EventBuild


def serialize_events(
        selected_model_events: Dict[int, EventInterface],
        path
):
    dic = {

        event_idx: {
            "Score": float(selected_model_events[event_idx].score),
            "BDC": float(selected_model_events[event_idx].bdc),
            "Position Array": selected_model_events[event_idx].pos_array.tolist(),
            "Point Array": selected_model_events[event_idx].point_array.tolist(),
            "Size": float(selected_model_events[event_idx].size),
            "Centroid": [
                    float(selected_model_events[event_idx].centroid[0]),
                    float(selected_model_events[event_idx].centroid[1]),
                    float(selected_model_events[event_idx].centroid[2]),
                             ],
            "Local Strength": float(selected_model_events[event_idx].local_strength),
            "Build": {
                "Build Path": str(selected_model_events[event_idx].build.build_path),
                "Ligand Key": selected_model_events[event_idx].build.ligand_key,
                "Score": float(selected_model_events[event_idx].build.score),
                "Centroid": [
                    float(selected_model_events[event_idx].build.centroid[0]),
                    float(selected_model_events[event_idx].build.centroid[1]),
                    float(selected_model_events[event_idx].build.centroid[2]),
                             ],
                "BDC": float(selected_model_events[event_idx].build.bdc),
                "Build Score" : float(selected_model_events[event_idx].build.build_score),
                "Noise": float(selected_model_events[event_idx].build.noise),
                "Signal": float(selected_model_events[event_idx].build.signal),
                "Num. Contacts": int(selected_model_events[event_idx].build.num_contacts),
                "Num. Points": float(selected_model_events[event_idx].build.num_points),
                'Optimal Contour': float(selected_model_events[event_idx].build.optimal_contour),
                'RSCC': float(selected_model_events[event_idx].build.rscc),
            }
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
            size=event['Size'],
            centroid=np.array(event["Centroid"]),
            score=event["Score"],
            bdc=event["BDC"],
            build=EventBuild(
                build_path=event["Build"]["Build Path"],
                ligand_key=event["Build"]["Ligand Key"],
                score=event["Build"]["Score"],
                centroid=np.array(event["Build"]["Centroid"]),
                bdc=event["Build"]["BDC"],
                build_score=event["Build"]["Build Score"],
                noise=event["Build"]["Noise"],
                signal=event["Build"]["Signal"],
                num_contacts=event["Build"]['Num. Contacts'],
                num_points=event["Build"]['Num. Points'],
                optimal_contour=event["Build"]['Optimal Contour'],
                rscc=event["Build"]['RSCC'],
            ),
            local_strength=event['Local Strength']
        )
        for event_idx, event
        in dic.items()
    }
