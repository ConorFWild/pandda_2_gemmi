from pandda_gemmi.analyse_interface import *

def get_reference_compatability_filtered_datasets(console, filter_reference_compatability, datasets_smoother, reference):
    console.start_reference_comparability_filters()
    datasets_reference: DatasetsInterface = filter_reference_compatability(datasets_smoother, reference)
    datasets: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                   datasets_reference.items()}
    console.summarise_filtered_datasets(
        filter_reference_compatability.filtered_dtags
    )
    return datasets