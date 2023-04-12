import os
import re
import subprocess
import inspect
from pathlib import Path

from .. import constants
from .autobuild import AutobuildResult
from ..fs import try_remove


def test_function():
    ...


class Rhofit:

    def __init__(self, cut=2.0):
        self.cut = cut

    def __call__(self, dmap_path, mtz_path, model_path, cif_path, out_dir):
        # Make rhofit commands
        pandda_rhofit_script_file = Path(
            os.path.dirname(inspect.getfile(test_function))).resolve() / constants.PANDDA_RHOFIT_SCRIPT_FILE
        print(f"pandda_rhofit_script_file: {pandda_rhofit_script_file}")
        print(f"Pandda Rhofit path: {Path(os.path.dirname(__file__)).resolve() / constants.PANDDA_RHOFIT_SCRIPT_FILE}")

        rhofit_command: str = constants.RHOFIT_COMMAND.format(
            pandda_rhofit=str(pandda_rhofit_script_file),
            event_map=str(dmap_path),
            mtz=str(mtz_path),
            pdb=str(model_path),
            cif=str(cif_path),
            out_dir=str(out_dir),
            cut=self.cut,
        )
        print(f"Command: {rhofit_command}")

        # Execute job script
        self.execute(rhofit_command)

        # Parse the log file
        log_path = out_dir / constants.RHOFIT_OUTPUT_DIR / constants.RHOFIT_CORR_FILE
        if log_path.exists():
            with open(log_path, "r") as f:
                log_info = f.read()

            log_results = re.findall(constants.RHOFIT_LOG_REGEX, log_info)

            log_result_dict = {}
            for log_result in log_results:
                file_path = out_dir / log_result[0]
                rscc = log_result[1]
                log_result_dict[str(file_path)] = rscc

            #
            try_remove(out_dir / constants.RHOFIT_OUTPUT_DIR / constants.RHOFIT_REFINE_MTZ)

            return AutobuildResult(
                log_result_dict,
                dmap_path,
                mtz_path,
                model_path,
                cif_path,
                out_dir
            )

        else:
            return AutobuildResult(
                None,
                dmap_path,
                mtz_path,
                model_path,
                cif_path,
                out_dir
            )

        # # Parse the results
        # for model_path in out_dir.glob(constants.RHOFIT_MODEL_REGEX):

    def execute(self, command: str):
        p = subprocess.Popen(command,
                             shell=True,
                             env=os.environ.copy(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             )

        stdout, stderr = p.communicate()
        print(str(stdout))
        print(str(stderr))
