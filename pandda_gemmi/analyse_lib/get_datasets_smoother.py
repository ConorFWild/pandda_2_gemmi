import time

from pandda_gemmi.analyse_interface import *
from pandda_gemmi.common import Partial


def get_datasets_smoother(console, smooth_func, process_local, datasets_wilson, reference, structure_factors, pandda_log):
    console.start_b_factor_smoothing()

    # with STDOUTManager('Performing b-factor smoothing...', f'\tDone!'):
    start = time.time()
    datasets_smoother: DatasetsInterface = {
        smoothed_dtag: smoothed_dataset
        for smoothed_dtag, smoothed_dataset
        in zip(
            datasets_wilson,
            process_local(
                [
                    Partial(smooth_func).paramaterise(
                        dataset,
                        reference,
                        structure_factors,
                    )
                    for dtag, dataset
                    in datasets_wilson.items()
                ]
            )
        )
    }

    finish = time.time()
    pandda_log["Time to perform b factor smoothing"] = finish - start

    console.summarise_b_factor_smoothing(datasets_smoother)

    return datasets_smoother