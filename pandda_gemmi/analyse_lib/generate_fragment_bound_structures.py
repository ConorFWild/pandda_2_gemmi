from pandda_gemmi.analyse_interface import *
from pandda_gemmi import constants
from pandda_gemmi.common import update_log
from pandda_gemmi.autobuild import (
    merge_ligand_into_structure_from_paths,
    save_pdb_file,
)


def generate_fragment_bound_structures(
        pandda_args,
        pandda_fs_model,
        datasets,
        autobuild_results,
        event_scores,
        console,
        pandda_log
):
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
            for path in autobuild_result.scores:
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
