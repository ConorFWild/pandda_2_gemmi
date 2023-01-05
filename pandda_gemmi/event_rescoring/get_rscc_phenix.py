import re
import subprocess
import pathlib

MATCH_REGEX = "([^\s]+)\s+LIG\s+([^\s]+)\s+([^\s]+)"

PHENIX_MODEL_MAP_CC_SCRIPT = (
    "cd {tmp_dir}; "
    "module load phenix; "
    "phenix.model_map_cc "
    "{model_path} "
    "{map_path} "
    "resolution={resolution} "
    "ignore_symmetry_conflicts=True "
    "wrapping=True ")


def get_rscc(
        dataset_bound_state_model_path,
        event_map_path,
        resolution,
        tmp_dir,
):
    # Make phenix command
    script = PHENIX_MODEL_MAP_CC_SCRIPT.format(
        tmp_dir=tmp_dir,
        model_path=dataset_bound_state_model_path,
        map_path=event_map_path,
        resolution=resolution,
    )

    # Run it
    p = subprocess.Popen(
        script,
        shell=True
    )
    p.communicate()

    # parse results
    results_path = pathlib.Path(tmp_dir) / "cc_per_residue.log"

    results = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            for line in f.readlines():
                # print(line)
                match = re.match(MATCH_REGEX, str(line))
                if not match:
                    continue
                chain = match.groups()[0]
                res = int(match.groups()[1])
                rscc = float(match.groups()[2])
                results[(chain, res)] = rscc

        return results

    else:
        return None