import os
from pathlib import Path

import fire
import gemmi
import numpy as np

from pandda_gemmi import constants
from pandda_gemmi.dmaps import save_dmap, load_dmap


def generate_event_maps(pandda_dir, dtag):
    processed_dataset_dir = Path(pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR
    dataset_dir = processed_dataset_dir / dtag
    all_event_map_dir = dataset_dir / "all_event_maps"
    if not all_event_map_dir.exists():
        os.mkdir(all_event_map_dir)

    dataset_map = dataset_dir / "xmap.ccp4"
    mean_map_path = dataset_dir / constants.PANDDA_MEAN_MAP_FILE.format(dtag=dtag)

    dataset_map = load_dmap(dataset_map)
    dataset_map_array = np.array(dataset_map, copy=False)

    mean_map = load_dmap(mean_map_path)
    mean_map_array = np.array(mean_map, copy=False)

    for fraction in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
        event_map_path = all_event_map_dir / f"{fraction}.ccp4"
        calc_event_map_array = (dataset_map_array - (1 - fraction) * mean_map_array) / fraction

        event_map = gemmi.FloatGrid(
            dataset_map.nu, dataset_map.nv, dataset_map.nw
        )
        event_map_array = np.array(event_map, copy=False)
        event_map_array[:, :, :] = calc_event_map_array[:, :, :]
        event_map.set_unit_cell(dataset_map.unit_cell)
        save_dmap(
            event_map,
            event_map_path
        )


if __name__ == "__main__":
    fire.Fire(generate_event_maps)
