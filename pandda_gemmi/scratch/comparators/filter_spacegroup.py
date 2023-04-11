from ..interfaces import *

class FilterSpaceGroup:

    def __init__(self, reference_dataset: DatasetInterface):
        self.reference_dataset = reference_dataset
    def __call__(self,
                 datasets: Dict[str, DatasetInterface],
                 ) :
        same_spacegroup_datasets = filter(
            lambda dtag: datasets[dtag].reflections.spacegroup() == self.reference_dataset.reflections.spacegroup(),
            datasets,
        )

        new_datasets = {dtag: datasets[dtag] for dtag in same_spacegroup_datasets}

        return new_datasets