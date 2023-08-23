from pathlib import Path

from pandda_gemmi.analyse import process_pandda
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.pandda_logging import PanDDAConsole

console = PanDDAConsole()

if __name__ == "__main__":
    console.start_pandda()

    # Parse Command Line Arguments
    console.start_parse_command_line_args()
    args = PanDDAArgs(
        data_dirs=Path("/dls/science/groups/i04-1/conor_dev/baz2b_test/data"),
        out_dir=Path(
            "/dls/labxchem/data/2017/lb18145-17/processing/analysis/pandda_2/pandda_2_test_runs/2023_01_05_inbuilt"),
        pdb_regex="*.dimple.pdb",
        mtz_regex="*.dimple.mtz"
    )
    console.summarise_arguments(args)

    # Process the PanDDA
    process_pandda(args)
