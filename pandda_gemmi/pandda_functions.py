from __future__ import annotations

from typing import *

import numpy as np
import joblib

from pandda_gemmi.pandda_types import *


def process_local_joblib(n_jobs, verbosity, funcs):
    mapper = joblib.Parallel(n_jobs=n_jobs,
                             verbose=verbosity,
                             backend="loky",
                             )

    results = mapper(joblib.delayed(func)() for func in funcs)

    return results


def get_comparators_high_res_random(
        datasets: Dict[Dtag, Dataset],
        comparison_min_comparators,
        comparison_max_comparators,
):
    dtag_list = [dtag for dtag in datasets]

    dtags_by_res = list(
        sorted(
            dtag_list,
            key=lambda dtag: datasets[dtag].reflections.resolution().resolution,
        )
    )

    highest_res_datasets = dtags_by_res[:comparison_min_comparators]
    highest_res_datasets_max = max([datasets[dtag].reflections.resolution().resolution for dtag in highest_res_datasets])

    comparators = {}
    for dtag in dtag_list:

        current_res = datasets[dtag].reflections.resolution().resolution

        truncation_res = max(current_res, highest_res_datasets_max)

        truncated_datasets = [dtag for dtag in dtag_list if datasets[dtag].reflections.resolution().resolution < truncation_res]

        comparators[dtag] = list(np.random.choice(truncated_datasets))

    return comparators