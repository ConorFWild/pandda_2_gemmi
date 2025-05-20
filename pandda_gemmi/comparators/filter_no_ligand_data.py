import re

from ..interfaces import *

class FilterNoLigandData:
    def exclude(self, dtag, datasets):
        dataset = datasets[dtag]

        # Skip processing the dataset if there is no ligand data
        if len([_key for _key in dataset.ligand_files if dataset.ligand_files[_key].ligand_cif]) == 0:
            return False
        else:
            return True


    def __call__(self, datasets: Dict[str, DatasetInterface]):

        good_rfree_dtags = filter(
            lambda _dtag: self.exclude(_dtag, datasets),
            datasets,
        )

        new_datasets = {dtag: datasets[dtag] for dtag in good_rfree_dtags}

        return new_datasets

    def description(self):
        return f"Filtered because no ligand data!"

