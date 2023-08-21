import yaml
from pandda_gemmi.autobuild import AutobuildResult
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
                    if ligand_autobuild_results
                }

            }

    with open(path, 'w') as f:
        yaml.dump(dic, f, sort_keys=False)

def unserialize_autobuilds(path):
    with open(path, 'r') as f:
        dic = yaml.safe_load( f)

    event_autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]] = {}

    for dtag, dtag_results in dic.items():
        for event_idx, event_results in dtag_results.items():
            event_autobuilds[(dtag, int(event_idx))] ={}

            for ligand_key, ligand_results in event_results["Ligand Autobuild Results"].items():
                event_autobuilds[(dtag, int(event_idx))][ligand_key] = AutobuildResult(
                    log_result_dict=ligand_results,
                    dmap_path=None,
                    mtz_path=None,
                    model_path=None,
                    cif_path=None,
                    out_dir=None
                )
