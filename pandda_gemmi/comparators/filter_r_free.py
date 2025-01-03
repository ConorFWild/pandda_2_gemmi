from ..interfaces import *


class FilterRFree:
    def __init__(self, max_rfree: float):
        self.max_rfree = max_rfree

    def __call__(self, datasets: Dict[str, DatasetInterface]):
        good_rfree_dtags = filter(
            lambda dtag: datasets[dtag].structure.rfree() < self.max_rfree,
            datasets,
        )

        new_datasets = {dtag: datasets[dtag] for dtag in good_rfree_dtags}


        return new_datasets

    def description(self):
        return f"Filtered because rfree > {self.max_rfree}"