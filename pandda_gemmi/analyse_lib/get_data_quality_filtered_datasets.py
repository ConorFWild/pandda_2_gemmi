from pandda_gemmi.analyse_interface import *

def get_data_quality_filtered_datasets(console, filter_data_quality, datasets_initial, structure_factors):
    console.start_data_quality_filters()
    datasets_for_filtering: DatasetsInterface = {dtag: dataset for dtag, dataset in
                                                 datasets_initial.items()}
    datasets_quality_filtered: DatasetsInterface = filter_data_quality(datasets_for_filtering, structure_factors)
    console.summarise_filtered_datasets(
        filter_data_quality.filtered_dtags
    )
    return datasets_quality_filtered