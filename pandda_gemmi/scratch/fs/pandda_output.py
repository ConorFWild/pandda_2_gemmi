import os
from pathlib import Path

from .pandda_input import PanDDAInput
from .. import constants

def try_make(path):
    try:
        os.mkdir(path)
    except:
        return

class PanDDAOutput:
    def __init__(self, path: Path, pandda_input: PanDDAInput):
        self.path = path
        self.processed_datasets_dir = path / constants.PANDDA_PROCESSED_DATASETS_DIR
        try_make(self.processed_datasets_dir)

        self.processed_datasets = {}
        for dtag, dataset_dir in pandda_input.dataset_dirs.items():
            processed_dataset_dir = self.processed_datasets_dir / dtag
            try_make(processed_dataset_dir)
            self.processed_datasets[dtag] = processed_dataset_dir
