from ..interfaces import *

class FilterResolutionLowerLimit:
    def __init__(self, high_res_limit_lower: float):
        self.high_res_limit_lower = high_res_limit_lower

    def __call__(self, datasets: Dict[str, DatasetInterface]):
        good_res_dtags = filter(
            lambda dtag: datasets[dtag].reflections.resolution() < self.high_res_limit_lower,
            datasets,
        )

        new_datasets = {dtag: datasets[dtag] for dtag in good_res_dtags}

        return new_datasets

    def description(self):
        return f"Filtered because resolution > {self.high_res_limit_lower}"
