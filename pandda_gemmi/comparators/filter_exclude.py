import re

from ..interfaces import *

class FilterExcludeFromAnalysis:
    def __init__(self, exclude_string: str):
        self.exclude_string = exclude_string
        self.datasets_to_exclude = [x for x in self.exclude_string.split(",")]

    def exclude(self, dtag):
        if dtag in self.datasets_to_exclude:
            return False
        else:
            return True


    def __call__(self, datasets: Dict[str, DatasetInterface]):

        good_rfree_dtags = filter(
            lambda dtag: self.exclude(dtag),
            datasets,
        )

        new_datasets = {dtag: datasets[dtag] for dtag in good_rfree_dtags}

        return new_datasets

    def description(self):
        return f"Filtered because in exclude set!"