from ..interfaces import *


class FilterResolution:


    def __init__(self,
                 dataset_resolution: float,
                 hard_min_num_datasets: int,
                 soft_min_num_datasets: int,
                 buffer: float):
        self.dataset_resolution = dataset_resolution
        self.hard_min_num_datasets = hard_min_num_datasets
        self.soft_min_num_datasets = soft_min_num_datasets
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
            # If the dataset is of the appropriate resolution, add it
            if dataset_res < self.dataset_resolution:
                new_datasets[dtag] = dataset
                _k = _k + 1
            # If it is not of the appropriate resolution, and the hard limit has not been reached, still add it
            elif _k < self.hard_min_num_datasets:
                new_datasets[dtag] = dataset
                _k = _k + 1
            # If if it not of the appropriate resolution, and the hard limit has been reached, add it if it is of
            # A reasonable resolution AND in the soft limit
            elif (_k < (self.soft_min_num_datasets)) and (dataset_res < (self.dataset_resolution + self.buffer)):
                new_datasets[dtag] = dataset
                _k = _k + 1
            # if (_k < self.min_num_datasets) or (dataset_res < (self.dataset_resolution + self.buffer)):
            #     new_datasets[dtag] = dataset
            #     _k = _k + 1


        return new_datasets
