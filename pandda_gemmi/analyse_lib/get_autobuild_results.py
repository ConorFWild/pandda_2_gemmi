import time

from pandda_gemmi.common import update_log, Partial
from pandda_gemmi.analyse_interface import *
from pandda_gemmi.autobuild import (
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
)
from pandda_gemmi import constants


def get_autobuild_results(pandda_args, console, process_local,
                          process_global, autobuild_func, pandda_fs_model, datasets, all_events, shell_results, pandda_log, event_scores,):
    autobuild_results = {}
    if autobuild_func:
        console.start_autobuilding()

        if pandda_args.global_processing == 'serial':
            process_autobuilds = process_local
        else:
            # process_autobuilds = process_global
            process_autobuilds = process_global

        time_autobuild_start = time.time()

        if autobuild_func.tag == "rhofit":
            autobuild_results: AutobuildResultsInterface = {
                event_id: autobuild_result
                for event_id, autobuild_result
                in zip(
                    all_events,
                    process_autobuilds(
                        [
                            Partial(autobuild_func).paramaterise(
                                datasets[event_id.dtag],
                                all_events[event_id],
                                pandda_fs_model,
                                cif_strategy=pandda_args.cif_strategy,
                                cut=2.0,
                                rhofit_coord=pandda_args.rhofit_coord,
                                debug=pandda_args.debug,
                            )
                            for event_id
                            in all_events
                        ],
                    )
                )
            }
        elif autobuild_func.tag == "inbuilt":
            event_scoring_results = {}
            for res, shell_result in shell_results.items():
                if shell_result:
                    for dtag, dataset_result in shell_result.dataset_results.items():
                        for event_id, event_scoring_result in dataset_result.event_scores.items():
                            event_scoring_results[event_id] = event_scoring_result

            autobuild_results: AutobuildResultsInterface = {
                event_id: autobuild_result
                for event_id, autobuild_result
                in zip(
                    all_events,
                    process_autobuilds(
                        [
                            Partial(autobuild_func).paramaterise(
                                event_id,
                                datasets[event_id.dtag],
                                all_events[event_id],
                                pandda_fs_model,
                                event_scoring_results[event_id],
                                debug=pandda_args.debug,
                            )
                            for event_id
                            in all_events
                        ],
                    )
                )
            }

        time_autobuild_finish = time.time()
        pandda_log[constants.LOG_AUTOBUILD_TIME] = time_autobuild_finish - time_autobuild_start

        # Save results
        pandda_log[constants.LOG_AUTOBUILD_COMMANDS] = {}
        pandda_log["autobuild_scores"] = {}

        for event_id, autobuild_result in autobuild_results.items():
            dtag = str(event_id.dtag)
            if dtag not in pandda_log[constants.LOG_AUTOBUILD_COMMANDS]:
                pandda_log[constants.LOG_AUTOBUILD_COMMANDS][dtag] = {}

            event_idx = int(event_id.event_idx.event_idx)

            pandda_log[constants.LOG_AUTOBUILD_COMMANDS][dtag][event_idx] = autobuild_result.log()

            if dtag not in pandda_log["autobuild_scores"]:
                pandda_log["autobuild_scores"][dtag] = {}
            pandda_log["autobuild_scores"][dtag][event_idx] = autobuild_result.scores

        console.summarise_autobuilding(autobuild_results)

        # with STDOUTManager('Updating the PanDDA models with best scoring fragment build...', f'\tDone!'):
        console.start_autobuild_model_update()
        # Add the best fragment by scoring method to default model
        pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILDS] = {}
        pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILD_SCORES] = {}
        autobuild_to_event = {}
        dataset_selected_events = {}
        for dtag in datasets:
            dataset_autobuild_results: AutobuildResultsInterface = {
                event_id: autobuild_result
                for event_id, autobuild_result
                in autobuild_results.items()
                if dtag == event_id.dtag
            }

            if len(dataset_autobuild_results) == 0:
                # print("\tNo autobuilds for this dataset!")
                continue

            all_scores = {}
            for event_id, autobuild_result in dataset_autobuild_results.items():
                for path, score in autobuild_result.scores.items():
                    # all_scores[path] = score
                    all_scores[path] = event_scores[event_id]
                    autobuild_to_event[path] = (dtag.dtag, event_id.event_idx.event_idx, path)

            if len(all_scores) == 0:
                # print(f"\tNo autobuilds for this dataset!")
                continue

            # Select fragment build
            selected_fragement_path = max(
                all_scores,
                key=lambda _path: all_scores[_path],
            )

            dataset_selected_events[dtag.dtag] = autobuild_to_event[selected_fragement_path]

            pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILDS][str(dtag)] = str(selected_fragement_path)
            pandda_log[constants.LOG_AUTOBUILD_SELECTED_BUILD_SCORES][str(dtag)] = float(
                all_scores[selected_fragement_path])

            # Copy to pandda models
            model_path = str(pandda_fs_model.processed_datasets.processed_datasets[dtag].input_pdb)
            pandda_model_path = pandda_fs_model.processed_datasets.processed_datasets[
                                    dtag].dataset_models.path / constants.PANDDA_EVENT_MODEL.format(str(dtag))
            merged_structure = merge_ligand_into_structure_from_paths(model_path, selected_fragement_path)
            save_pdb_file(merged_structure, pandda_model_path)

        console.summarise_autobuild_model_update(dataset_selected_events)

        update_log(pandda_log, pandda_args.out_dir / constants.PANDDA_LOG_FILE)

        console.summarise_autobuilds(autobuild_results)

    return autobuild_results