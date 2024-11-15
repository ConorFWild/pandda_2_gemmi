import re

from ..interfaces import *

class FilterOnlyDatasets:
    def __init__(self, only_string: str):
        self.only_string = only_string
        if only_string is not None:
            self.datasets_to_include = [x for x in self.only_string.split(",")]
        else:
            self.datasets_to_include = None

    def exclude(self, dtag):
        if self.datasets_to_include is None:
            return True
        else:
            if dtag in self.datasets_to_include:
                return True
            else:
                return False


    def __call__(self, datasets: Dict[str, DatasetInterface]):

        good_rfree_dtags = filter(
            lambda dtag: self.exclude(dtag),
            datasets,
        )

        new_datasets = {dtag: datasets[dtag] for dtag in good_rfree_dtags}

        return new_datasets

    def description(self):
        return f"Filtered because in exclude set!"