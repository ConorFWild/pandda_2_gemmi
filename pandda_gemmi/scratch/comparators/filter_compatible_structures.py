import gemmi

from ..interfaces import *


class FilterCompatibleStructures:
    def __init__(self, dataset, similarity=0.9):
        self.dataset = dataset
        self.similarity = similarity

    def get_compatible(self, dataset):
        ref_mod = self.dataset.structure.structure[0]
        ref_mod_sequence = ref_mod.full_sequence

        mov_mod = dataset.structure.structre[0]
        mov_mod_chains = [chain.name for chain in mov_mod]

        for chain in ref_mod:
            if chain.name not in mov_mod_chains:
                return False
            else:
                result = gemmi.align_sequence_to_polymer(
                    ref_mod_sequence,
                    chain.get_polymer(),
                    gemmi.PolymerType.PeptideL,
                )

                if not result.calculate_identity() > self.similarity:
                    return False

        return True

    def __call__(self, datasets: Dict[str, DatasetInterface]):
        new_datasets = {}
        for _dtag in datasets:
            dataset = datasets[_dtag]
            if self.get_compatible(dataset):
                new_datasets[_dtag] = dataset

        return new_datasets
