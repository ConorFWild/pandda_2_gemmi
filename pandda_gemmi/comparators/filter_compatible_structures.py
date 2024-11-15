from ..interfaces import *


class FilterCompatibleStructures:
    def __init__(self, dataset, similarity=100.0):
        self.dataset = dataset
        self.similarity = similarity
        self.resids = set(resid for resid in self.dataset.structure.protein_residue_ids())


    def get_compatible(self, dataset):
        dataset_resids = set([resid for resid in dataset.structure.protein_residue_ids()])
        if len(dataset_resids.intersection(self.resids)) != len(self.resids):
            return False
        else:
            return True

    def __call__(self, datasets: Dict[str, DatasetInterface]):
        new_datasets = {}
        for _dtag in datasets:
            dataset = datasets[_dtag]
            if self.get_compatible(dataset):
                new_datasets[_dtag] = dataset


        return new_datasets
