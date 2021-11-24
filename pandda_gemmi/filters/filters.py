from pandda_gemmi.dataset import Reference, Datasets
from pandda_gemmi.edalignment import Alignment


def remove_models_with_large_gaps(datasets, reference: Reference):
    new_dtags = filter(lambda dtag: Alignment.has_large_gap(reference, datasets.datasets[dtag]),
                       datasets.datasets,
                       )

    new_datasets = {dtag: datasets.datasets[dtag] for dtag in new_dtags}

    return Datasets(new_datasets)
