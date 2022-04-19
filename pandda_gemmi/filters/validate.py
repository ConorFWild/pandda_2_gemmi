from pandda_gemmi.analyse_interface import *


class DatasetsValidator(DatasetsValidatorInterface):

    def __init__(self, min_characterisation_datasets: int):
        self.min_characterisation_datasets = min_characterisation_datasets

    def __call__(self, datasets: DatasetsInterface, exception: str):
        if len(datasets) < self.min_characterisation_datasets:
            raise Exception(exception)
