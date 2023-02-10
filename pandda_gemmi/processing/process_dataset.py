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

# console = PanDDAConsole()
#
# printer = pprint.PrettyPrinter()

# Scientific python libraries
# import ray

## Custom Imports
from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.pandda_functions import (
    process_local_serial,
    truncate,
    save_native_frame_zmap,
    save_reference_frame_zmap,
)
from pandda_gemmi.common import Dtag, EventID, Partial, update_log, cache, uncache
from pandda_gemmi.edalignment import Partitioning, Xmap, XmapArray, Grid, from_unaligned_dataset_c, GetMapStatistics
from pandda_gemmi.event import (
    Event, Clusterings, Clustering, Events, get_event_mask_indicies,
    save_event_map,
)


@dataclasses.dataclass()
class DatasetResult(DatasetResultInterface):
    dtag: DtagInterface
    events: MutableMapping[EventIDInterface, EventInterface]
    event_scores: EventScoringResultsInterface
    log: Dict


class ModelSelection(ModelSelectionInterface):
    def __init__(self, selected_model_id: ModelIDInterface, log: Dict) -> None:
        self.selected_model_id = selected_model_id
        self.log = log


def EXPERIMENTAL_select_model(
        model_results: ModelResultsInterface,
        inner_mask: CrystallographicGridInterface,
        processed_dataset: ProcessedDatasetInterface,
        debug: Debug = Debug.DEFAULT,
) -> ModelSelectionInterface:
    log = {}

    model_event_scores: Dict[ModelIDInterface, Dict[EventIDInterface, EventScoringResultInterface]] = {
        model_id: model.event_scores for
        model_id,
        model in model_results.items()
    }

    if debug >= Debug.PRINT_NUMERICS:
        print("Event scores for each model are:")
        print(model_event_scores)

        for model_id, event_scores in model_event_scores.items():
            print(f"Best score for model: {model_id}")
            print(
                [
                    event_scores[score_id].get_selected_structure_score()
                    for score_id
                    in event_scores
                ]
            )
            for event_id, event_score_result in event_scores.items():
                print(f"event log: {event_id.event_idx.event_idx} {event_id.dtag.dtag}")
                print(event_score_result.log())

    # Score the top clusters#
    model_scores = {}
    for model_id, event_scores in model_event_scores.items():
        selected_event_scores = [
            event_scores[event_id].get_selected_structure_score()
            for event_id
            in event_scores
        ]
        if debug >= Debug.PRINT_NUMERICS:
            print(f"\tModel {model_id} all scores: {selected_event_scores}")

        filtered_model_scores = [
                                    selected_event_score
                                    for selected_event_score
                                    in selected_event_scores
                                    if selected_event_score
                                ] + [-0.001, ]
        if debug >= Debug.PRINT_NUMERICS:
            print(f"\tModel {model_id}: filtered scores: {filtered_model_scores}")

        maximum_event_score = max(
            filtered_model_scores
        )
        model_scores[model_id] = maximum_event_score

    # model_scores = {
    #     model_id: max(
    #         [
    #             event_scores[score_id].get_selected_structure_score()
    #             for score_id
    #             in event_scores
    #             if event_scores[score_id].get_selected_structure_score() is not None
    #         ] + [0.0, ])
    #     for model_id, event_scores
    #     in model_event_scores.items()
    # }

    if debug >= Debug.PRINT_SUMMARIES:
        print("Maximum score of any event for each model are are:")
        print(model_scores)

    log['model_scores'] = {
        model_id: float(score)
        for model_id, score
        in model_scores.items()
    }

    if len(model_scores) == 0:
        return ModelSelection(0, log)

    else:
        selected_model_number = max(
            model_scores,
            key=lambda _score: model_scores[_score],
        )  # [0]

    return ModelSelection(selected_model_number, log)


class SelectModel:

    def __call__(self, model_results: ModelResultsInterface, ):
        return EXPERIMENTAL_select_model(
            model_results,
        )


def process_dataset_multiple_models(
        test_dtag: DtagInterface,
        models: ModelsInterface,
        shell: ShellInterface,
        dataset_truncated_datasets: DatasetsInterface,
        alignments: AlignmentsInterface,
        dataset_xmaps: XmapsInterface,
        pandda_fs_model: PanDDAFSModelInterface,
        reference: ReferenceInterface,
        grid: GridInterface,
        contour_level: float,
        cluster_cutoff_distance_multiplier: float,
        min_blob_volume: float,
        min_blob_z_peak: float,
        structure_factors: StructureFactorsInterface,
        outer_mask: float,
        inner_mask_symmetry: float,
        max_site_distance_cutoff: float,
        min_bdc: float,
        max_bdc: float,
        sample_rate: float,
        statmaps: bool,
        analyse_model_func: AnalyseModelInterface,
        score_events_func: GetEventScoreInterface,
        process_local: ProcessorInterface,
        debug: Debug = Debug.DEFAULT,
) -> DatasetResultInterface:
    if debug >= Debug.PRINT_SUMMARIES:
        print(f'\tProcessing dtag: {test_dtag}')
    time_dataset_start = time.time()

    dataset_log_path = pandda_fs_model.processed_datasets.processed_datasets[test_dtag].log_path
    dataset_log = {}
    dataset_log["Model analysis time"] = {}

    ###################################################################
    # # Process the models...
    ###################################################################
    time_model_analysis_start = time.time()

    model_results: ModelResultsInterface = {
        model_number: model_result
        for model_number, model_result
        in zip(
            models,
            process_local(
                [
                    Partial(
                        analyse_model_func).paramaterise(
                        model,
                        model_number,
                        test_dtag=test_dtag,
                        dataset=dataset_truncated_datasets[test_dtag],
                        dataset_xmap=dataset_xmaps[test_dtag],
                        reference=reference,
                        grid=grid,
                        dataset_processed_dataset=pandda_fs_model.processed_datasets.processed_datasets[test_dtag],
                        dataset_alignment=alignments[test_dtag],
                        max_site_distance_cutoff=max_site_distance_cutoff,
                        min_bdc=min_bdc, max_bdc=max_bdc,
                        contour_level=contour_level,
                        cluster_cutoff_distance_multiplier=cluster_cutoff_distance_multiplier,
                        min_blob_volume=min_blob_volume,
                        min_blob_z_peak=min_blob_z_peak,
                        output_dir=pandda_fs_model.processed_datasets.processed_datasets[test_dtag].path,
                        score_events_func=score_events_func,
                        res=shell.res,
                        rate=0.5,
                        debug=debug
                    )
                    for model_number, model
                    in models.items()
                ]
            )
        )
    }

    dataset_log["Model logs"] = {
        model_number: model_result.model_log
        for model_number, model_result
        in model_results.items()
    }  #

    time_model_analysis_finish = time.time()

    dataset_log["Time to analyse all models"] = time_model_analysis_finish - time_model_analysis_start

    if debug >= Debug.PRINT_SUMMARIES:
        print(f"\tTime to analyse all models: {time_model_analysis_finish - time_model_analysis_start}")
        for model_number, model_result in model_results.items():
            model_time = dataset_log["Model logs"][model_number]["Model analysis time"]
            print(f"\t\tModel {model_number} processed in {model_time}")

    ###################################################################
    # # Decide which model to use...
    ###################################################################
    if debug >= Debug.PRINT_SUMMARIES:
        print(f"\tSelecting model...")
    model_selection: ModelSelectionInterface = EXPERIMENTAL_select_model(
        model_results,
        grid.partitioning.inner_mask,
        pandda_fs_model.processed_datasets.processed_datasets[test_dtag],
        debug=debug,
    )
    selected_model: ModelInterface = models[model_selection.selected_model_id]
    selected_model_clusterings = model_results[model_selection.selected_model_id].clusterings_merged
    zmap = uncache(model_results[model_selection.selected_model_id].zmap)
    dataset_log['Selected model'] = int(model_selection.selected_model_id)
    dataset_log['Model selection log'] = model_selection.log

    if debug >= Debug.PRINT_SUMMARIES:
        print(f'\tSelected model is: {model_selection.selected_model_id}')

    ###################################################################
    # # Output the z map
    ###################################################################
    time_output_zmap_start = time.time()

    native_grid = dataset_truncated_datasets[test_dtag].reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        # sample_rate=sample_rate,  # TODO: make this d_min/0.5?
        sample_rate=dataset_truncated_datasets[test_dtag].reflections.get_resolution() / 0.5
    )

    partitioning = Partitioning.from_structure_multiprocess(
        dataset_truncated_datasets[test_dtag].structure,
        native_grid,
        outer_mask,
        inner_mask_symmetry,
    )
    # pandda_fs_model.processed_datasets.processed_datasets[dtag].z_map_file.save_reference_frame_zmap(zmap)

    save_native_frame_zmap(
        pandda_fs_model.processed_datasets.processed_datasets[test_dtag].z_map_file.path,
        zmap,
        dataset_truncated_datasets[test_dtag],
        alignments[test_dtag],
        grid,
        structure_factors,
        outer_mask,
        inner_mask_symmetry,
        partitioning,
        sample_rate,
    )

    # TODO: Remove altogether
    if debug >= Debug.DATASET_MAPS:
        for model_number, model_result in model_results.items():
            save_reference_frame_zmap(
                pandda_fs_model.processed_datasets.processed_datasets[
                    test_dtag].z_map_file.path.parent / f'{model_number}_ref.ccp4',
                model_result.zmap
            )
            save_native_frame_zmap(
                pandda_fs_model.processed_datasets.processed_datasets[
                    test_dtag].z_map_file.path.parent / f'{model_number}_native.ccp4',
                model_result.zmap,
                dataset_truncated_datasets[test_dtag],
                alignments[test_dtag],
                grid,
                structure_factors,
                outer_mask,
                inner_mask_symmetry,
                partitioning,
                sample_rate,
            )

    # if statmaps:
    #     mean_map_file = MeanMapFile.from_zmap_file(
    #         pandda_fs_model.processed_datasets.processed_datasets[test_dtag].z_map_file)
    #     mean_map_file.save_native_frame_mean_map(
    #         selected_model,
    #         zmap,
    #         dataset_truncated_datasets[test_dtag],
    #         alignments[test_dtag],
    #         grid,
    #         structure_factors,
    #         outer_mask,
    #         inner_mask_symmetry,
    #         partitioning,
    #         sample_rate,
    #     )

    #     std_map_file = StdMapFile.from_zmap_file(pandda_fs_model.processed_datasets.processed_datasets[
    #                                                  test_dtag].z_map_file)
    #     std_map_file.save_native_frame_std_map(
    #         test_dtag,
    #         selected_model,
    #         zmap,
    #         dataset_truncated_datasets[test_dtag],
    #         alignments[test_dtag],
    #         grid,
    #         structure_factors,
    #         outer_mask,
    #         inner_mask_symmetry,
    #         partitioning,
    #         sample_rate,
    #     )
    time_output_zmap_finish = time.time()
    dataset_log['Time to output z map'] = time_output_zmap_finish - time_output_zmap_start

    ###################################################################
    # # Find the events
    ###################################################################
    time_event_start = time.time()
    # Calculate the shell events
    # events: Events = Events.from_clusters(
    #     selected_model_clusterings,
    #     selected_model,
    #     dataset_xmaps,
    #     grid,
    #     alignments[test_dtag],
    #     max_site_distance_cutoff,
    #     min_bdc, max_bdc,
    #     None,
    # )
    events = model_results[model_selection.selected_model_id].events

    time_event_finish = time.time()
    dataset_log["Number of events"] = len(events)
    dataset_log[constants.LOG_DATASET_EVENT_TIME] = time_event_finish - time_event_start
    update_log(dataset_log, dataset_log_path)

    ###################################################################
    # # Generate event maps
    ###################################################################
    time_event_map_start = time.time()

    # Save the event maps!
    # printer.pprint(events)
    Events(events).save_event_maps(
        dataset_truncated_datasets,
        alignments,
        dataset_xmaps,
        selected_model,
        pandda_fs_model,
        grid,
        structure_factors,
        outer_mask,
        inner_mask_symmetry,
        sample_rate,
        native_grid,
        mapper=ProcessLocalSerial(),
    )

    if debug >= Debug.DATASET_MAPS:
        for model_number, model_result in model_results.items():
            for event_id, event in model_result.events.items():
                save_event_map(
                    pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].path / f'{model_number}'
                                                                                                f'_{event_id.event_idx.event_idx}.ccp4',
                    dataset_xmaps[event_id.dtag],
                    models[model_number],
                    event,
                    dataset_truncated_datasets[event_id.dtag],
                    alignments[event_id.dtag],
                    grid,
                    structure_factors,
                    outer_mask,
                    inner_mask_symmetry,
                    partitioning,
                    sample_rate,
                )

    time_event_map_finish = time.time()
    dataset_log[constants.LOG_DATASET_EVENT_MAP_TIME] = time_event_map_finish - time_event_map_start
    update_log(dataset_log, dataset_log_path)

    time_dataset_finish = time.time()
    dataset_log[constants.LOG_DATASET_TIME] = time_dataset_finish - time_dataset_start
    update_log(dataset_log, dataset_log_path)

    return DatasetResult(
        dtag=test_dtag,
        events={event_id: event for event_id, event in events.items()},
        event_scores=model_results[model_selection.selected_model_id].event_scores,
        log=dataset_log,
    )
