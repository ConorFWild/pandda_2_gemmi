from typing import *
from pandda_gemmi.analyse_interface import *

from pandda_gemmi.common import Dtag
from pandda_gemmi.dataset import Dataset


class DatasetsStatistics(DatasetsStatisticsInterface):
    def __init__(self, datasets: DatasetsInterface):
        self.unit_cells = DatasetsStatistics.get_unit_cell_stats(datasets)
        self.spacegroups = DatasetsStatistics.get_spacegroup_stats(datasets)
        self.resolutions = DatasetsStatistics.get_resolution_stats(datasets)
        self.chains = DatasetsStatistics.get_chain_stats(datasets)
        # self.residues = ...

    @staticmethod
    def get_unit_cell_stats(datasets: Dict[Dtag, Dataset]):

        unit_cells = [dataset.reflections.reflections.cell for dtag, dataset in datasets.items()]

        unit_cells = {"a": [unit_cell.a for unit_cell in unit_cells],
         "b": [unit_cell.b for unit_cell in unit_cells],
         "c": [unit_cell.c for unit_cell in unit_cells],
         "alpha": [unit_cell.alpha for unit_cell in unit_cells],
         "beta": [unit_cell.beta for unit_cell in unit_cells],
         "gamma": [unit_cell.gamma for unit_cell in unit_cells],
         }

        return unit_cells

    @staticmethod
    def get_spacegroup_stats(datasets: Dict[Dtag, Dataset]):
        spacegroups = [dataset.reflections.reflections.spacegroup.hm for dtag, dataset in datasets.items()]

        return spacegroups

    @staticmethod
    def get_resolution_stats(datasets: Dict[Dtag, Dataset]):
        resolutions = [dataset.reflections.reflections.resolution_high() for dtag, dataset in datasets.items()]
        return resolutions

    @staticmethod
    def get_chain_stats(datasets: Dict[Dtag, Dataset]):
        chains = []
        for dtag, dataset in datasets.items():

            dataset_structure = dataset.structure.structure

            dataset_chains = []

            for model in dataset_structure:
                for chain in model:
                    dataset_chains.append(chain.name)

            dataset_chains = list(sorted(dataset_chains))

            chains.append(dataset_chains)

        return chains