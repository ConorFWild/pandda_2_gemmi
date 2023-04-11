import os
from pathlib import Path

from .pandda_input import PanDDAInput
from .. import constants


class PanDDAOutput:
    def __init__(self, path: Path, pandda_input: PanDDAInput):
        self.path = path
        self.processed_datasets_dir = path / constants.PANDDA_PROCESSED_DATASETS_DIR
        os.mkdir(self.processed_datasets_dir)
        self.processed_datasets = {}
        for dtag, dataset_dir in pandda_input.dataset_dirs.items():
            processed_dataset_dir = dataset_dir.path / dtag
            os.mkdir(processed_dataset_dir)
            self.processed_datasets[dtag] = processed_dataset_dir
