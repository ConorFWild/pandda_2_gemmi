from pathlib import Path

import fire
import gemmi

from pandda_gemmi import constants

def generate_event_maps(pandda_dir, dtag):
    processed_dataset_dir = Path(pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR
    dataset_dir = processed_dataset_dir / dtag

    dataset_map = dataset_dir / "xmap.ccp4"
    mean_map_path = dataset_dir / constants.PANDDA_MEAN_MAP_FILE.format(dtag=dtag)

    dataset_map =
    dataset_map_array

    mean_map
    mean_map_array

    for fraction in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
        event_map_path = dataset_dir / f"{fraction}.ccp4"
        event_map_array =


if __name__ == "__main__":
    fire.Fire(generate_event_maps)