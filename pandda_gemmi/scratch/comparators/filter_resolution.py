from ..interfaces import *


class FilterResolution:
    def __init__(self, resolution_cutoff: float):
        self.resolution_cutoff = resolution_cutoff

    def __call__(self, datasets: Dict[str, DatasetInterface], ):
        high_resolution_dtags = filter(
            lambda dtag: datasets[dtag].reflections.resolution() < self.resolution_cutoff,
            datasets,
        )

        new_datasets = {dtag: datasets[dtag] for dtag in high_resolution_dtags}

        return new_datasets
