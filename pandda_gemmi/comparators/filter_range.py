import re

from ..interfaces import *

class FilterRange:
    def __init__(self, dataset_range: str):
        self.dataset_range = dataset_range


        self.range_min , self.range_max = self.parse_range(dataset_range)

    @staticmethod
    def parse_range(dataset_range):
        match = re.match(
            "([0-9]+)-([0-9]+)",
            dataset_range
        )
        if not match:
            raise Exception
        groups = match.groups()
        if not groups:
            raise Exception
        if not len(groups) == 2:
            raise Exception

        return int(groups[0]), int(groups[1])


    @staticmethod
    def get_dtag_idx(dtag):
        matches = re.findall(
            "[0-9]+",
            dtag
        )
        return int(matches[-1])

    def in_range(self, dtag):
        idx = self.get_dtag_idx(dtag)
        if (idx >= self.range_min) & (idx < self.range_max):
            return True
        else:
            return False


    def __call__(self, datasets: Dict[str, DatasetInterface]):
        good_rfree_dtags = filter(
            lambda dtag: self.in_range(dtag),
            datasets,
        )

        new_datasets = {dtag: datasets[dtag] for dtag in good_rfree_dtags}


        return new_datasets

    def description(self):
        return f"Filtered because outside range {self.range_min}-{self.range_max}"