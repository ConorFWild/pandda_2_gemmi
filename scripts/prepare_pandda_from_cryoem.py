from pathlib import Path

import fire

from pandda_gemmi.cryoem.mrc_to_mtz import mrc_to_mtz


def prepare_pandda_from_cryoem(input_dir, output_dir, mrc_regex="*.mrc"):
    # Get the input and output dir
    input_dir, output_dir = Path(input_dir), Path(output_dir)

    # Iterate over directories, converting mrcs to mtzs
    for dataset_dir in input_dir.glob("*"):
        # Get the input mrc file
        mrcs = [x for x in dataset_dir.glob(mrc_regex)]

        if len(mrcs) != 1:
            print(f"Input dataset directory {dataset_dir} does not contain an mrc file! Exiting!")

        mrc_file = mrcs[0]

        #
        mtz_file = dataset_dir / f"dimple.mtz"

        mrc_to_mtz(
            mrc_file,
            mtz_file
        )


if __name__ == '__main__':
    fire.Fire()
