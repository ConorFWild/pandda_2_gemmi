# Base python
import dataclasses
import time
import pprint
from functools import partial
import os
import json
from typing import Set
import pickle

from pandda_gemmi.processing.process_local import ProcessLocalSerial
from pandda_gemmi.pandda_logging import STDOUTManager, log_arguments, PanDDAConsole
from pandda_gemmi.processing.process_dataset import process_dataset_multiple_models

console = PanDDAConsole()

printer = pprint.PrettyPrinter()

# Scientific python libraries
# import ray
import numpy as np
import gemmi

## Custom Imports
from pandda_gemmi.logs import (
    summarise_array,
)

from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.pandda_functions import (
    process_local_serial,
    truncate,
    save_native_frame_zmap,
    save_reference_frame_zmap,
)
from pandda_gemmi.python_types import *
from pandda_gemmi.common import Dtag, EventID, Partial
from pandda_gemmi.fs import PanDDAFSModel, MeanMapFile, StdMapFile
from pandda_gemmi.dataset import (StructureFactors, Dataset, Datasets,
                                  Resolution, )
from pandda_gemmi.shells import Shell, ShellMultipleModels
from pandda_gemmi.edalignment import Partitioning, Xmap, XmapArray, Grid, from_unaligned_dataset_c, GetMapStatistics
from pandda_gemmi.model import Zmap, Model, Zmaps
from pandda_gemmi.event import (
    Event, Clusterings, Clustering, Events, get_event_mask_indicies,
    save_event_map,
)
from pandda_gemmi.density_clustering import (
    GetEDClustering, FilterEDClusteringsSize,
    FilterEDClusteringsPeak,
    MergeEDClusterings,
)

def get_models(
        test_dtags: List[DtagInterface],
        comparison_sets: Dict[int, List[DtagInterface]],
        shell_xmaps: XmapsInterface,
        grid: GridInterface,
        process_local: ProcessorInterface,
):
    masked_xmap_array = XmapArray.from_xmaps(
        shell_xmaps,
        grid,
    )

    models = {}
    for comparison_set_id, comparison_set_dtags in comparison_sets.items():
        # comparison_set_dtags =

        # Get the relevant dtags' xmaps
        masked_train_characterisation_xmap_array: XmapArray = masked_xmap_array.from_dtags(
            comparison_set_dtags)
        masked_train_all_xmap_array: XmapArray = masked_xmap_array.from_dtags(
            comparison_set_dtags + [test_dtag for test_dtag in test_dtags])

        mean_array: np.ndarray = Model.mean_from_xmap_array(masked_train_characterisation_xmap_array,
                                                            )  # Size of grid.partitioning.total_mask > 0
        # dataset_log[constants.LOG_DATASET_MEAN] = summarise_array(mean_array)
        # update_log(dataset_log, dataset_log_path)

        sigma_is: Dict[Dtag, float] = Model.sigma_is_from_xmap_array(masked_train_all_xmap_array,
                                                                     mean_array,
                                                                     1.5,
                                                                     )  # size of n
        # dataset_log[constants.LOG_DATASET_SIGMA_I] = {_dtag.dtag: float(sigma_i) for _dtag, sigma_i in sigma_is.items()}
        # update_log(dataset_log, dataset_log_path)

        sigma_s_m: np.ndarray = Model.sigma_sms_from_xmaps(masked_train_characterisation_xmap_array,
                                                           mean_array,
                                                           sigma_is,
                                                           process_local,
                                                           )  # size of total_mask > 0
        # dataset_log[constants.LOG_DATASET_SIGMA_S] = summarise_array(sigma_s_m)
        # update_log(dataset_log, dataset_log_path)

        model: ModelInterface = Model.from_mean_is_sms(
            mean_array,
            sigma_is,
            sigma_s_m,
            grid,
        )
        models[comparison_set_id] = model

    return models