import time

from pandda_gemmi.common import update_log, Partial
from pandda_gemmi.analyse_interface import *
from pandda_gemmi.autobuild import (
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
)
from pandda_gemmi import constants


def get_shell_results_async(pandda_args,
                            console,
                            process_global,
                            process_local,
                            pandda_fs_model,
                            process_shell_function,
                            load_xmap_func, analyse_model_func, score_events_func, shells, structure_factors, datasets,
                            reference, alignments, grid, pandda_log, ):
    console.start_process_shells()

    # Process the shells
    time_shells_start = time.time()
    # if pandda_args.comparison_strategy == "cluster" or pandda_args.comparison_strategy == "hybrid":

    for res, shell in shells.items():
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
        xmaps = get_xmaps(console, pandda_fs_model, process_local_in_shell, load_xmap_func, structure_factors,
                          alignments,
                          grid, shell, shell_truncated_datasets, sample_rate, shell_log, debug, shell_log_path)

        ###################################################################
        # # Get the models to test
        ###################################################################
        models = get_shell_models(console, pandda_fs_model, process_local_in_shell, shell, grid, xmaps, debug)

        ###################################################################
        # # Process each test dataset
        ###################################################################


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
