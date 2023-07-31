import yaml

from pandda_gemmi.interfaces import *


def processed_autobuilds(
        datasets: Dict[str, DatasetInterface],
        event_autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]],
        path
):
    dic = {}

    for dtag in sorted(datasets):
        dtag_autobuilds: Dict[int, Dict[str, AutobuildInterface]] = {
            _event_idx: event_autobuilds[(dtag, _event_idx)]
            for _event_idx
            in sorted([x[1] for x in event_autobuilds if x[0] == dtag])
        }

        if len(dtag_autobuilds) > 0:
            dic[dtag] = {}

        for event_idx, autobuild in dtag_autobuilds.items():
            dic[dtag][event_idx] = {
                "Ligand Autobuild Results": {
                    ligand_key: {
                        str(conformer_model_path): round(
                            ligand_autobuild_results.log_result_dict[conformer_model_path],
                            2,
                        )
                        for conformer_model_path
                        in sorted(
                            ligand_autobuild_results.log_result_dict,
                            key=lambda _x: ligand_autobuild_results.log_result_dict[_x])
                    }
                    for ligand_key, ligand_autobuild_results
                    in autobuild.items()
                }

            }

    with open(path, 'w') as f:
        yaml.dump(dic, f)
