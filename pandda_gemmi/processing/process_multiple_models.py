# Base python
import dataclasses
import time
import pprint
import os
import json

from pandda_gemmi.processing.process_local import ProcessLocalSerial
from pandda_gemmi.pandda_logging import STDOUTManager, log_arguments, PanDDAConsole
from pandda_gemmi.processing.process_dataset import process_dataset_multiple_models
from pandda_gemmi.model.get_models import get_models

# Scientific python libraries
# import ray
import numpy as np
import gemmi

## Custom Imports
from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.pandda_functions import (
    truncate,
)
from pandda_gemmi.python_types import *
from pandda_gemmi.common import Dtag, EventID, Partial
from pandda_gemmi.dataset import (StructureFactors, Dataset, Datasets,
                                  Resolution, )
from pandda_gemmi.edalignment import save_xmap, save_raw_xmap, save_array_to_map_file





@dataclasses.dataclass()
class ShellResult(ShellResultInterface):
    shell: ShellInterface
    dataset_results: DatasetResultsInterface
    log: Dict


def update_log(shell_log, shell_log_path):
    if shell_log_path.exists():
        os.remove(shell_log_path)

    with open(shell_log_path, "w") as f:
        json.dump(shell_log, f, indent=4)


def process_shell_multiple_models(
        shell: ShellInterface,
        datasets: DatasetsInterface,
        alignments: AlignmentsInterface,
        grid: GridInterface,
        pandda_fs_model: PanDDAFSModelInterface,
        reference: ReferenceInterface,
        process_local: ProcessorInterface,
        structure_factors: StructureFactorsInterface,
        sample_rate: float,
        contour_level: float,
        cluster_cutoff_distance_multiplier: float,
        min_blob_volume: float,
        min_blob_z_peak: float,
        outer_mask: float,
        inner_mask_symmetry: float,
        max_site_distance_cutoff: float,
        min_bdc: float,
        max_bdc: float,
        memory_availability: str,
        statmaps: bool,
        load_xmap_func: LoadXMapInterface,
        analyse_model_func: AnalyseModelInterface,
        score_events_func: GetEventScoreInterface,
        debug: Debug = Debug.DEFAULT,
):
    console = PanDDAConsole()
    printer = pprint.PrettyPrinter()

    if debug >= Debug.DEFAULT:
        console.print_starting_process_shell(shell)

    process_local_in_dataset, process_local_in_shell, process_local_over_datasets = get_processors(process_local,
                                                                                                   memory_availability)

    time_shell_start = time.time()
    if pandda_fs_model.shell_dirs:
        shell_log_path = pandda_fs_model.shell_dirs.shell_dirs[shell.res].log_path
    else:
        raise Exception(
            "Attempted to find the log path for the shell, but no shell dir added to pandda_fs_model somehow.")
    shell_log = {}

    # Seperate out test and train datasets
    shell_datasets: DatasetsInterface = {
        dtag: dataset
        for dtag, dataset
        in datasets.items()
        if dtag in shell.all_dtags
    }
    shell_log[constants.LOG_SHELL_DATASETS] = [dtag.dtag for dtag in shell_datasets]
    update_log(shell_log, shell_log_path)

    ###################################################################
    # # Homogonise shell datasets by truncation of resolution
    ###################################################################
    shell_truncated_datasets = get_shell_datasets(console, datasets, structure_factors, shell, shell_datasets,
                                                  shell_log, debug)

    ###################################################################
    # # Generate aligned Xmaps
    ###################################################################
    xmaps = get_xmaps(console, pandda_fs_model, process_local_in_shell, load_xmap_func, structure_factors, alignments,
                      grid, shell, shell_truncated_datasets, sample_rate, shell_log, debug, shell_log_path)

    ###################################################################
    # # Get the models to test
    ###################################################################
    models = get_shell_models(console, pandda_fs_model, process_local_in_shell, shell, grid, xmaps, debug)

    ###################################################################
    # # Process each test dataset
    ###################################################################
    results = process_shell_datasets(console, pandda_fs_model, analyse_model_func, process_local_in_dataset,
                                     process_local_over_datasets, score_events_func, models, structure_factors,
                                     reference, alignments, grid, shell_truncated_datasets, xmaps, statmaps, shell,
                                     outer_mask, min_blob_z_peak, min_blob_volume, cluster_cutoff_distance_multiplier,
                                     inner_mask_symmetry, max_bdc, max_site_distance_cutoff, contour_level, min_bdc,
                                     debug)

    # Update shell log with dataset results
    shell_log[constants.LOG_SHELL_DATASET_LOGS] = {}
    for result in results:
        if result:
            shell_log[constants.LOG_SHELL_DATASET_LOGS][str(result.dtag)] = result.log

    time_shell_finish = time.time()
    shell_log[constants.LOG_SHELL_TIME] = time_shell_finish - time_shell_start
    update_log(shell_log, shell_log_path)

    shell_result: ShellResultInterface = ShellResult(
        shell=shell,
        dataset_results={dtag: result for dtag, result in zip(shell.test_dtags, results) if result},
        log=shell_log,

    )

    if debug >= Debug.DEFAULT:
        console.print_summarise_process_datasets(shell_result)

    return shell_result


def process_shell_datasets(console, pandda_fs_model, analyse_model_func, process_local_in_dataset,
                           process_local_over_datasets, score_events_func, models, structure_factors, reference,
                           alignments, grid, shell_truncated_datasets, xmaps, statmaps, shell, outer_mask,
                           min_blob_z_peak, min_blob_volume, cluster_cutoff_distance_multiplier, inner_mask_symmetry,
                           max_bdc, max_site_distance_cutoff, contour_level, min_bdc, debug):
    # Now that all the data is loaded, get the comparison set and process each test dtag
    # process_dataset_paramaterized =
    if debug >= Debug.DEFAULT:
        console.print_starting_process_datasets()
    # Process each dataset in the shell
    all_train_dtags_unmerged = [_dtag for l in shell.train_dtags.values() for _dtag in l]
    all_train_dtags = []
    for _dtag in all_train_dtags_unmerged:
        if _dtag not in all_train_dtags:
            all_train_dtags.append(_dtag)
    if debug >= Debug.PRINT_NUMERICS:
        print(f"\tAll train datasets are: {all_train_dtags}")
    # dataset_dtags = {_dtag:  for _dtag in shell.test_dtags for n in shell.train_dtags}
    dataset_dtags = {_dtag: [_dtag] + all_train_dtags for _dtag in shell.test_dtags}
    if debug >= Debug.PRINT_NUMERICS:
        print(f"\tDataset dtags are: {dataset_dtags}")
    results: List[DatasetResultInterface] = process_local_over_datasets(
        [
            Partial(
                process_dataset_multiple_models).paramaterise(
                test_dtag,
                # dataset_truncated_datasets={_dtag: shell_truncated_datasets[_dtag] for _dtag in
                #                             dataset_dtags[test_dtag]},
                dataset_truncated_datasets={test_dtag: shell_truncated_datasets[test_dtag]},
                # dataset_xmaps={_dtag: xmaps[_dtag] for _dtag in dataset_dtags[test_dtag]},
                dataset_xmaps={test_dtag: xmaps[test_dtag], },
                models=models,
                shell=shell,
                alignments=alignments,
                pandda_fs_model=pandda_fs_model,
                reference=reference,
                grid=grid,
                contour_level=contour_level,
                cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
                min_blob_volume=min_blob_volume,
                min_blob_z_peak=min_blob_z_peak,
                structure_factors=structure_factors,
                outer_mask=outer_mask,
                inner_mask_symmetry=inner_mask_symmetry,
                max_site_distance_cutoff=max_site_distance_cutoff,
                min_bdc=min_bdc,
                max_bdc=max_bdc,
                # sample_rate=sample_rate,
                sample_rate=shell.res / 0.5,
                statmaps=statmaps,
                analyse_model_func=analyse_model_func,
                score_events_func=score_events_func,
                process_local=process_local_in_dataset,
                debug=debug,
            )

            for test_dtag
            in shell.test_dtags
        ],
    )
    for result in results:
        if result:
            result_dtag = result.dtag
            for model_number, model_log in result.log["Model logs"].items():
                precent_outlying_protein = model_log["ZMap statistics"]["Percent Outlying Protein"]
                print(
                    f"Dataset {result_dtag.dtag} model {model_number} percentage outlying protein {precent_outlying_protein}")
    return results


def get_shell_models(console, pandda_fs_model, process_local_in_shell, shell, grid, xmaps, debug):
    if debug >= Debug.DEFAULT:
        console.print_starting_get_models()
    time_models_start = time.time()
    models: ModelsInterface = get_models(
        shell.test_dtags,
        shell.train_dtags,
        xmaps,
        grid,
        process_local_in_shell,
    )
    if debug >= Debug.PRINT_SUMMARIES:
        for model_key, model in models.items():
            save_array_to_map_file(
                model.mean,
                grid.grid,
                pandda_fs_model.pandda_dir / f"{shell.res}_{model_key}_mean.ccp4"
            )
    time_models_finish = time.time()
    if debug >= Debug.DEFAULT:
        console.print_summarise_get_models(models, time_models_finish - time_models_start)
    return models


def get_xmaps(console, pandda_fs_model, process_local_in_shell, load_xmap_func, structure_factors, alignments, grid,
              shell, shell_truncated_datasets, sample_rate, shell_log, debug, shell_log_path):
    if debug >= Debug.DEFAULT:
        console.print_starting_loading_xmaps()
    time_xmaps_start = time.time()
    xmaps: XmapsInterface = {
        dtag: xmap
        for dtag, xmap
        in zip(
            shell_truncated_datasets,
            process_local_in_shell(
                [
                    Partial(load_xmap_func).paramaterise(
                        shell_truncated_datasets[key],
                        alignments[key],
                        grid=grid,
                        structure_factors=structure_factors,
                        sample_rate=shell.res / 0.5,
                    )
                    for key
                    in shell_truncated_datasets
                ]
            )
        )
    }
    time_xmaps_finish = time.time()
    shell_log[constants.LOG_SHELL_XMAP_TIME] = time_xmaps_finish - time_xmaps_start
    update_log(shell_log, shell_log_path)
    if debug >= Debug.DATASET_MAPS:
        for dtag, xmap in xmaps.items():
            xmap_array = np.array(xmap.xmap)
            save_array_to_map_file(
                xmap_array,
                grid.grid,
                pandda_fs_model.pandda_dir / f"{shell.res}_{dtag}_ref.ccp4"
            )

            save_raw_xmap(
                shell_truncated_datasets[dtag],
                pandda_fs_model.pandda_dir / f"{shell.res}_{dtag}_mov.ccp4",
                structure_factors,
                sample_rate
            )

            # save_xmap(
            #     xmap,
            #     pandda_fs_model.pandda_dir / f"{shell.res}_{dtag}.ccp4"
            # )
    if debug >= Debug.DEFAULT:
        console.print_summarise_loading_xmaps(xmaps, time_xmaps_finish - time_xmaps_start)
    return xmaps


def get_shell_datasets(console, datasets, structure_factors, shell, shell_datasets, shell_log, debug):
    if debug >= Debug.DEFAULT:
        console.print_starting_truncating_shells()
    shell_working_resolution: ResolutionInterface = Resolution(
        max([datasets[dtag].reflections.get_resolution() for dtag in shell.all_dtags]))
    shell_truncated_datasets: DatasetsInterface = truncate(
        shell_datasets,
        resolution=shell_working_resolution,
        structure_factors=structure_factors,
    )
    # TODO: REMOVE?
    # shell_truncated_datasets = shell_datasets
    shell_log["Shell Working Resolution"] = shell_working_resolution.resolution
    if debug >= Debug.DEFAULT:
        console.print_summarise_truncating_shells(shell_truncated_datasets)
    return shell_truncated_datasets


def get_processors(process_local, memory_availability):
    if memory_availability == "very_low":
        process_local_in_shell: ProcessorInterface = ProcessLocalSerial()
        process_local_in_dataset: ProcessorInterface = ProcessLocalSerial()
        process_local_over_datasets: ProcessorInterface = ProcessLocalSerial()
    elif memory_availability == "low":
        process_local_in_shell: ProcessorInterface = process_local
        process_local_in_dataset: ProcessorInterface = process_local
        process_local_over_datasets: ProcessorInterface = ProcessLocalSerial()
    elif memory_availability == "high":
        process_local_in_shell: ProcessorInterface = process_local
        process_local_in_dataset: ProcessorInterface = ProcessLocalSerial()
        process_local_over_datasets: ProcessorInterface = process_local

    else:
        raise Exception(f"memory_availability: {memory_availability}: does not have defined processors")
    return process_local_in_dataset, process_local_in_shell, process_local_over_datasets
