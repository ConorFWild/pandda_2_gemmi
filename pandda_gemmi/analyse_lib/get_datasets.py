from pandda_gemmi.dataset import GetDatasets
from pandda_gemmi.dataset import DatasetsStatistics
from pandda_gemmi.analyse_interface import *


def get_datasets(pandda_args, console, pandda_fs_model):
    console.start_load_datasets()
    datasets_initial: DatasetsInterface = GetDatasets()(pandda_fs_model, )
    datasets_statistics: DatasetsStatisticsInterface = DatasetsStatistics(datasets_initial)
    console.summarise_datasets(datasets_initial, datasets_statistics)

    if pandda_args.debug >= Debug.PRINT_NUMERICS:
        print(datasets_initial)

    return datasets_initial, datasets_statistics
