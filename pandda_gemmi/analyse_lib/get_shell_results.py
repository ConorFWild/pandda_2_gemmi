import time

from pandda_gemmi.common import update_log, Partial
from pandda_gemmi.analyse_interface import *
from pandda_gemmi.autobuild import (
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
)
from pandda_gemmi import constants

import time

from pandda_gemmi.common import update_log, Partial
from pandda_gemmi.analyse_interface import *
from pandda_gemmi.autobuild import (
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
)
from pandda_gemmi import constants
from pandda_gemmi.model import get_models
from pandda_gemmi.processing.process_multiple_models import get_xmaps, ShellResult, get_processors, get_shell_datasets
from pandda_gemmi.common import cache, uncache
from pandda_gemmi.processing.process_dataset import EXPERIMENTAL_select_model, DatasetResult
from pandda_gemmi.processing.process_local import ProcessLocalSerial
from pandda_gemmi.pandda_logging import PanDDAConsole
from pandda_gemmi.pandda_functions import (
    save_native_frame_zmap,
    save_reference_frame_zmap,
)
from pandda_gemmi.edalignment import Partitioning
from pandda_gemmi.event import (
    Events,
    save_event_map,
)


def merge_dataset_model_results(
        test_dtag: DtagInterface,
        model_results: ModelResultsInterface,
        shell_models: ModelsInterface,
        grid: GridInterface,
        dataset_xmaps: XmapsInterface,
        pandda_fs_model: PanDDAFSModelInterface,
        dataset_truncated_datasets: DatasetsInterface,
        structure_factors: StructureFactorsInterface,
        outer_mask,
        inner_mask_symmetry,
        alignments,
        sample_rate,
        debug: Debug,
) -> DatasetResultInterface:
    dataset_log = {}
    dataset_log_path = pandda_fs_model.processed_datasets.processed_datasets[test_dtag].log_path

    dataset_log["Model logs"] = {
        model_number: model_result.model_log
        for model_number, model_result
        in model_results.items()
    }  #

    time_model_analysis_finish = time.time()

    # dataset_log["Time to analyse all models"] = time_model_analysis_finish - time_model_analysis_start

    if debug >= Debug.PRINT_SUMMARIES:
        # print(f"\tTime to analyse all models: {time_model_analysis_finish - time_model_analysis_start}")
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
    selected_model: ModelInterface = shell_models[model_selection.selected_model_id]
    selected_model_clusterings = model_results[model_selection.selected_model_id].clusterings_merged
    zmap = model_results[model_selection.selected_model_id].zmap
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

    time_output_zmap_finish = time.time()
    dataset_log['Time to output z map'] = time_output_zmap_finish - time_output_zmap_start

    ###################################################################
    # # Find the events
    ###################################################################
    time_event_start = time.time()
    # Calculate the shell events

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
                    shell_models[model_number],
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
    # dataset_log[constants.LOG_DATASET_TIME] = time_dataset_finish - time_dataset_start
    update_log(dataset_log, dataset_log_path)

    return DatasetResult(
        dtag=test_dtag,
        events={event_id: event for event_id, event in events.items()},
        event_scores=model_results[model_selection.selected_model_id].event_scores,
        log=dataset_log,
    )


def merge_shell_dataset_results(datasets_results: DatasetResultsInterface,
                                shell: ShellInterface,
                                pandda_fs_model: PanDDAFSModelInterface,
                                console,
                                debug: Debug) -> ShellResultInterface:
    shell_log = {}
    shell_log_path = pandda_fs_model.shell_dirs.shell_dirs[shell.res].log_path

    shell_log[constants.LOG_SHELL_DATASET_LOGS] = {}
    for dtag, result in datasets_results.items():
        if result:
            shell_log[constants.LOG_SHELL_DATASET_LOGS][str(result.dtag)] = result.log

    # time_shell_finish = time.time()
    # shell_log[constants.LOG_SHELL_TIME] = time_shell_finish - time_shell_start
    update_log(shell_log, shell_log_path)

    shell_result: ShellResultInterface = ShellResult(
        shell=shell,
        dataset_results={dtag: result for dtag, result in datasets_results.items() if result},
        log=shell_log,

    )

    if debug >= Debug.DEFAULT:
        console.print_summarise_process_datasets(shell_result)

    return shell_result


def get_shell_results_async(
        pandda_args,
        console,
        process_global: ProcessorAsyncInterface,
        process_local: ProcessorInterface,
        pandda_fs_model: PanDDAFSModelInterface,
        load_xmap_func,
        analyse_model_func,
        score_events_func,
        shells: ShellsInterface,
        structure_factors,
        datasets: DatasetsInterface,
        reference: ReferenceInterface,
        alignments: AlignmentsInterface,
        grid: GridInterface,
        pandda_log,
):
    console.start_process_shells()

    # Process the shells
    time_shells_start = time.time()

    shell_dataset_model_futures = {}
    model_caches = {}
    shell_truncated_datasets_cache = {}
    shell_xmaps_chace = {}

    time_shell_submit_start = time.time()

    for res, shell in shells.items():
        console = PanDDAConsole()
        # printer = pprint.PrettyPrinter()

        if pandda_args.debug >= Debug.DEFAULT:
            console.print_starting_process_shell(shell)

        process_local_in_dataset, process_local_in_shell, process_local_over_datasets = get_processors(
            process_local,
            pandda_args.memory_availability,
        )

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
        shell_truncated_datasets = get_shell_datasets(
            console,
            datasets,
            structure_factors,
            shell,
            shell_datasets,
            shell_log,
            pandda_args.debug,
        )
        shell_truncated_datasets_cache[res] = cache(
            pandda_fs_model.shell_dirs.shell_dirs[res].path,
            shell_truncated_datasets,
        )

        ###################################################################
        # # Generate test Xmaps
        ###################################################################
        test_datasets_to_load = {_dtag: _dataset for _dtag, _dataset in shell_truncated_datasets.items() if
                                 _dtag in shell.test_dtags}

        test_xmaps = get_xmaps(
            console,
            pandda_fs_model,
            process_local_in_shell,
            load_xmap_func,
            structure_factors,
            alignments,
            grid,
            shell,
            test_datasets_to_load,
            pandda_args.sample_rate,
            shell_log,
            pandda_args.debug,
            shell_log_path,
        )
        shell_xmaps_chace[res] = cache(
            pandda_fs_model.shell_dirs.shell_dirs[res].path,
            test_xmaps,
        )
        for model_number, comparators in shell.train_dtags.items():
            time_model_start = time.time()
            ###################################################################
            # # Generate train Xmaps
            ###################################################################
            train_datasets_to_load = {_dtag: _dataset for _dtag, _dataset in shell_truncated_datasets.items() if
                                      _dtag in comparators}
            train_xmaps = get_xmaps(
                console,
                pandda_fs_model,
                process_local_in_shell,
                load_xmap_func,
                structure_factors,
                alignments,
                grid,
                shell,
                train_datasets_to_load,
                pandda_args.sample_rate,
                shell_log,
                pandda_args.debug,
                shell_log_path,
            )

            ###################################################################
            # # Get the model to test
            ###################################################################
            combined_xmaps = {}
            for dtag, xmap in test_xmaps.items():
                combined_xmaps[dtag] = xmap
            for dtag, xmap in train_xmaps.items():
                combined_xmaps[dtag] = xmap
            model: ModelInterface = get_models(
                shell.test_dtags,
                {model_number: comparators, },
                combined_xmaps,
                grid,
                process_local
            )[model_number]
            model_caches[(res, model_number)] = cache(
                pandda_fs_model.shell_dirs.shell_dirs[res].path,
                model,
            )

            ###################################################################
            # # Process each test dataset
            ###################################################################
            for test_dtag in shell.test_dtags:
                print(f"\t\t\tSubmitting: {test_dtag.dtag}")
                future_id = (res, test_dtag, model_number)
                shell_dataset_model_futures[future_id] = process_global.submit(
                    Partial(
                        analyse_model_func).paramaterise(
                        model,
                        model_number,
                        test_dtag=test_dtag,
                        dataset=shell_truncated_datasets[test_dtag],
                        dataset_xmap=test_xmaps[test_dtag],
                        reference=reference,
                        grid=grid,
                        dataset_processed_dataset=pandda_fs_model.processed_datasets.processed_datasets[test_dtag],
                        dataset_alignment=alignments[test_dtag],
                        max_site_distance_cutoff=pandda_args.max_site_distance_cutoff,
                        min_bdc=pandda_args.min_bdc, max_bdc=pandda_args.max_bdc,
                        contour_level=pandda_args.contour_level,
                        cluster_cutoff_distance_multiplier=pandda_args.cluster_cutoff_distance_multiplier,
                        min_blob_volume=pandda_args.min_blob_volume,
                        min_blob_z_peak=pandda_args.min_blob_z_peak,
                        output_dir=pandda_fs_model.processed_datasets.processed_datasets[test_dtag].path,
                        score_events_func=score_events_func,
                        res=shell.res,
                        rate=0.5,
                        debug=pandda_args.debug
                    )
                )

                # shell_dataset_model_futures[future_id] = process_global.submit(
                # Partial(
                #         analyse_model_func).paramaterise(
                #         model,
                #         model_number,
                #         test_dtag=test_dtag,
                #         dataset=shell_truncated_datasets[test_dtag],
                #         dataset_xmap=test_xmaps[test_dtag],
                #         reference=reference,
                #         grid=grid,
                #         dataset_processed_dataset=pandda_fs_model.processed_datasets.processed_datasets[test_dtag],
                #         dataset_alignment=alignments[test_dtag],
                #         max_site_distance_cutoff=pandda_args.max_site_distance_cutoff,
                #         min_bdc=pandda_args.min_bdc, max_bdc=pandda_args.max_bdc,
                #         contour_level=pandda_args.contour_level,
                #         cluster_cutoff_distance_multiplier=pandda_args.cluster_cutoff_distance_multiplier,
                #         min_blob_volume=pandda_args.min_blob_volume,
                #         min_blob_z_peak=pandda_args.min_blob_z_peak,
                #         output_dir=pandda_fs_model.processed_datasets.processed_datasets[test_dtag].path,
                #         score_events_func=score_events_func,
                #         res=shell.res,
                #         rate=0.5,
                #         debug=pandda_args.debug
                #     )()
            time_model_finish = time.time()
            print(f"\t\tProcessed model in {time_model_finish - time_model_start}")

        time_shell_finish = time.time()
        print(f"\tProcessed shell in {time_shell_finish - time_shell_start}")

    time_shell_submit_finish = time.time()
    ###################################################################
    # # Await the results...
    ###################################################################
    print(f"Submitted all shells in {time_shell_submit_finish-time_shell_submit_start}, awaiting results!")
    model_results = {_future_id: future.get() for _future_id, future in shell_dataset_model_futures.items()}
    print(f"Got all shell results!")

    ###################################################################
    # # Get the dataset results
    ###################################################################
    get_results_start = time.time()
    shell_results = {}
    for res, shell in shells.items():
        print(f"\tAssembing shell results for shell: {res}")
        time_shell_result_start = time.time()
        shell_dtag_results = {}
        shell_models = {model_cache_id[1]: uncache(model_cache_path) for model_cache_id, model_cache_path in
                        model_caches.items() if model_cache_id[0] == res}
        shell_truncated_datasets = uncache(shell_truncated_datasets_cache[res], remove=True)
        shell_xmaps = uncache(shell_xmaps_chace[res], remove=True)
        for dtag in shell.test_dtags:
            print(f"\t\tAssembling dtag results for dtag: {dtag.dtag}")
            model_assemble_start = time.time()
            dtag_model_results = {
                model_id[2]: model_result
                for model_id, model_result
                in model_results.items() if
                model_id[1] == dtag}
            dataset_result = merge_dataset_model_results(
                dtag,
                dtag_model_results,
                shell_models,
                grid,
                {dtag: shell_xmaps[dtag]},
                pandda_fs_model,
                {dtag: shell_truncated_datasets[dtag]},
                structure_factors,
                pandda_args.outer_mask,
                pandda_args.inner_mask_symmetry,
                alignments,
                pandda_args.sample_rate,
                pandda_args.debug,
            )
            model_assemble_finish = time.time()
            shell_dtag_results[dtag] = dataset_result
            print(f"\t\t\tGet dataset result in: {model_assemble_finish-model_assemble_start}")
        shell_result = merge_shell_dataset_results(
            shell_dtag_results,
            shell,
            pandda_fs_model,
            console,
            pandda_args.debug
        )
        time_shell_result_finish = time.time()
        print(f"\t\tGot shell result in {time_shell_result_finish-time_shell_result_start}")

        shell_results[res] = shell_result
    get_results_finish = time.time()
    print(f"Assembled all shell results in {get_results_finish - get_results_start}!")

    ###################################################################
    # # Get the model to test
    ###################################################################

    pandda_log[constants.LOG_SHELLS] = {
        res: shell_result.log
        for res, shell_result
        in shell_results.items()
        if shell_result
    }
    time_shells_finish = time.time()
    pandda_log["Time to process all shells"] = time_shells_finish - time_shells_start
    if pandda_args.debug >= Debug.PRINT_SUMMARIES:
        print(f"Time to process all shells: {time_shells_finish - time_shells_start}")

    all_events: EventsInterface = {}
    for res, shell_result in shell_results.items():
        if shell_result:
            for dtag, dataset_result in shell_result.dataset_results.items():
                all_events.update(dataset_result.events)

    event_scores: EventScoresInterface = {}
    for res, shell_result in shell_results.items():
        if shell_result:
            for dtag, dataset_result in shell_result.dataset_results.items():
                event_scores.update(
                    {
                        event_id: event_scoring_result.get_selected_structure_score()
                        for event_id, event_scoring_result
                        in dataset_result.event_scores.items()
                    }
                )

    # Add the event maps to the fs
    for event_id, event in all_events.items():
        pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].event_map_files.add_event(event)

    update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

    if pandda_args.debug >= Debug.PRINT_NUMERICS:
        print(shell_results)
        print(all_events)
        print(event_scores)

    console.summarise_shells(shell_results, all_events, event_scores)

    return shell_results, all_events, event_scores


def get_shell_results(pandda_args, console, process_global, process_local, pandda_fs_model, process_shell_function,
                      load_xmap_func, analyse_model_func, score_events_func, shells, structure_factors, datasets,
                      reference, alignments, grid, pandda_log, ):
    console.start_process_shells()

    # Process the shells
    time_shells_start = time.time()
    # if pandda_args.comparison_strategy == "cluster" or pandda_args.comparison_strategy == "hybrid":

    shell_results: ShellResultsInterface = {
        shell_id: shell_result
        for shell_id, shell_result
        in zip(
            shells,
            process_global(
                [
                    Partial(
                        process_shell_function).paramaterise(
                        shell,
                        datasets,
                        alignments,
                        grid,
                        pandda_fs_model,
                        reference,
                        process_local=process_local,
                        structure_factors=structure_factors,
                        sample_rate=pandda_args.sample_rate,
                        contour_level=pandda_args.contour_level,
                        cluster_cutoff_distance_multiplier=pandda_args.cluster_cutoff_distance_multiplier,
                        min_blob_volume=pandda_args.min_blob_volume,
                        min_blob_z_peak=pandda_args.min_blob_z_peak,
                        outer_mask=pandda_args.outer_mask,
                        inner_mask_symmetry=pandda_args.inner_mask_symmetry,
                        max_site_distance_cutoff=pandda_args.max_site_distance_cutoff,
                        min_bdc=pandda_args.min_bdc,
                        max_bdc=pandda_args.max_bdc,
                        memory_availability=pandda_args.memory_availability,
                        statmaps=pandda_args.statmaps,
                        load_xmap_func=load_xmap_func,
                        analyse_model_func=analyse_model_func,
                        score_events_func=score_events_func,
                        debug=pandda_args.debug,
                    )
                    for res, shell
                    in shells.items()
                ],
            )
        )
    }

    time_shells_finish = time.time()
    pandda_log[constants.LOG_SHELLS] = {
        res: shell_result.log
        for res, shell_result
        in shell_results.items()
        if shell_result
    }
    pandda_log["Time to process all shells"] = time_shells_finish - time_shells_start
    if pandda_args.debug >= Debug.PRINT_SUMMARIES:
        print(f"Time to process all shells: {time_shells_finish - time_shells_start}")

    all_events: EventsInterface = {}
    for res, shell_result in shell_results.items():
        if shell_result:
            for dtag, dataset_result in shell_result.dataset_results.items():
                all_events.update(dataset_result.events)

    event_scores: EventScoresInterface = {}
    for res, shell_result in shell_results.items():
        if shell_result:
            for dtag, dataset_result in shell_result.dataset_results.items():
                event_scores.update(
                    {
                        event_id: event_scoring_result.get_selected_structure_score()
                        for event_id, event_scoring_result
                        in dataset_result.event_scores.items()
                    }
                )

    # Add the event maps to the fs
    for event_id, event in all_events.items():
        pandda_fs_model.processed_datasets.processed_datasets[event_id.dtag].event_map_files.add_event(event)

    update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

    if pandda_args.debug >= Debug.PRINT_NUMERICS:
        print(shell_results)
        print(all_events)
        print(event_scores)

    console.summarise_shells(shell_results, all_events, event_scores)

    return shell_results, all_events, event_scores
