from pandda_gemmi.analyse_interface import *
from pandda_gemmi.fs import DatasetFile
def get_reference_compatability_filtered_datasets(console, filter_reference_compatability, datasets_smoother, reference, pandda_fs_model:PanDDAFSModelInterface):
    console.start_reference_comparability_filters()
    datasets_reference: DatasetsInterface = filter_reference_compatability(datasets_smoother, reference)
    datasets: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                   datasets_reference.items()}
    console.summarise_filtered_datasets(
        filter_reference_compatability.filtered_dtags
    )

    for dtag, dataset in datasets.items():
        pandda_fs_model.dataset_files[dtag].save(dataset)

    return datasets