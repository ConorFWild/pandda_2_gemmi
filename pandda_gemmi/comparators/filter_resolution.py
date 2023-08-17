from ..interfaces import *


class FilterResolution:


    def __init__(self, dataset_resolution: float, min_num_datasets: int, buffer: float):
        self.dataset_resolution = dataset_resolution
        self.min_num_datasets = min_num_datasets
        self.buffer=buffer


    def __call__(self, datasets: Dict[str, DatasetInterface], ):
        """
        Iterate through the datasets, from highest resolution to lowest, including them if either there are not already
        enough, or they are still better resolution than the reference (possibly + a buffer)
        :param datasets:
        :return:
        """

        new_datasets = {}
        _k = 0
        for dtag in sorted(
                datasets,
                key= lambda _dtag: datasets[_dtag].reflections.resolution()
        ):
            dataset = datasets[dtag]
            dataset_res = dataset.reflections.resolution()
            if (_k < self.min_num_datasets) or (dataset_res < (self.dataset_resolution + self.buffer)):
                new_datasets[dtag] = dataset
                _k = _k + 1


        return new_datasets
