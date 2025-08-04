import os
import shutil
import time
import inspect

# try:
#     from sklearnex import patch_sklearn
#
#     patch_sklearn()
# except ImportError:
#     print('No sklearn-express available!')

import gdown
import yaml

import numpy as np
import pandas as pd
import gemmi

from pandda_gemmi.interfaces import *
from pandda_gemmi.args import PanDDAArgs
from pandda_gemmi.fs import PanDDAFS
from pandda_gemmi.dataset import XRayDataset, StructureArray, Structure
from pandda_gemmi.dmaps import (
    SparseDMap,
    SparseDMapStream,
    TruncateReflections,
    SmoothReflections,
)
from pandda_gemmi.alignment import Alignment, DFrame
from pandda_gemmi.processor import ProcessLocalRay, Partial
from pandda_gemmi.comparators import (
    get_comparators,
    FilterRFree,
    FilterRange,
    FilterExcludeFromAnalysis,
    FilterOnlyDatasets,
    FilterSpaceGroup,
    FilterResolution,
    FilterCompatibleStructures,
    FilterResolutionLowerLimit,
    FilterNoLigandData
)
from pandda_gemmi.event_model.event import EventBuild
from pandda_gemmi.event_model.characterization import get_characterization_sets, CharacterizationNNAndFirst
from pandda_gemmi.event_model.filter_characterization_sets import filter_characterization_sets
from pandda_gemmi.event_model.outlier import PointwiseNormal, PointwiseMAD
from pandda_gemmi.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.event_model.score import get_model_map, ScoreCNNLigand
from pandda_gemmi.event_model.filter import (
    FilterSize,
    FilterScore,
    FilterSymmetryPosBuilds,
    FilterLocallyHighestBuildScoring
)
from pandda_gemmi.event_model.select import select_model
from pandda_gemmi.event_model.output import output_maps
from pandda_gemmi.event_model.filter_selected_events import filter_selected_events
from pandda_gemmi.event_model.get_bdc import get_bdc

from pandda_gemmi.site_model import HeirarchicalSiteModel, Site, get_sites
from pandda_gemmi.autobuild import AutobuildResult, ScoreCNNEventBuild
from pandda_gemmi.autobuild.inbuilt import mask_dmap, get_conformers, autobuild_conformer
from pandda_gemmi.autobuild.merge import merge_autobuilds, MergeHighestBuildScore
from pandda_gemmi.ranking import rank_events, RankHighEventScore, RankHighEventScoreBySite
from pandda_gemmi.tables import output_tables
from pandda_gemmi.pandda_logging import PanDDAConsole
from pandda_gemmi import serialize
from pandda_gemmi.cnn import load_model_from_checkpoint, EventScorer, LitEventScoring, BuildScorer, LitBuildScoring, \
    set_structure_mean

from pandda_gemmi.metrics import get_hit_in_site_probabilities
from pandda_gemmi.plots import plot_aligned_density_projection

def read_dataset(fs, dtag):
    pandda_events = {}
    autobuilds = {}
    processed_dataset_path = fs.output.processed_datasets[dtag] / f"events.yaml"
    if not processed_dataset_path.exists():
        return {}, {}
    _events = serialize.unserialize_events(processed_dataset_path)
    for event_idx, event in _events.items():
        pandda_events[(dtag, event_idx)] = event
    for event_idx, event in _events.items():
        autobuilds[(dtag, event_idx)] = {
            event.build.ligand_key: AutobuildResult(
                {event.build.build_path: {'score': event.build.signal, 'centroid': event.build.centroid}},
                None, None, None, None, None
            )
        }
    return pandda_events, autobuilds

def process_dataset(
        dtag,
        args,
        fs,
        datasets,
        console,
        j,
        datasets_to_process,
        time_begin_process_datasets,
        process_model,
        score_event,
        processor,
        dataset_refs,
        structure_array_refs,
        score_build_ref
):
    pandda_events = {}
    autobuilds = {}
    # Record the time that dataset processing begins
    time_begin_process_dataset = time.time()

    # Handle the case in which the dataset has already been processed
    # TODO: log properly
    events_yaml_path = fs.output.processed_datasets[dtag] / f"events.yaml"
    print(f"Checking for a event yaml at: {events_yaml_path}")
    if events_yaml_path.exists():
        print(f"Already have events for dataset! Skipping!")
        new_events, new_autobuilds = read_dataset(fs, dtag)
        pandda_events.update(new_events)
        autobuilds.update(new_autobuilds)

    # Get the dataset
    dataset = datasets[dtag]

    # Get the resolution of the dataset
    dataset_res = dataset.reflections.resolution()

    # Get the comparator datasets: these are filtered for reasonable data quality, space group compatability,
    # compatability of structural models and similar resolution
    comparator_datasets: Dict[str, DatasetInterface] = get_comparators(
        datasets,
        [
            FilterRFree(args.max_rfree),
            FilterSpaceGroup(dataset),
            FilterCompatibleStructures(dataset, debug=args.debug),
            FilterResolution(dataset_res, args.max_shell_datasets, 100, args.high_res_buffer)],
        debug=args.debug
    )

    # Ensure the dataset itself is included in comparators
    if dtag not in comparator_datasets:
        comparator_datasets[dtag] = dataset

    # Get the resolution to process the dataset at
    processing_res = max(
        [_dataset.reflections.resolution() for _dataset in comparator_datasets.values()]
    )

    # Print basic information about the processing to be done of the dataset
    console.begin_dataset_processing(
        dtag,
        dataset,
        dataset_res,
        comparator_datasets,
        processing_res,
        j,
        datasets_to_process,
        time_begin_process_datasets
    )

    # Skip if there are insufficient comparators in order to characterize a statistical model
    if len(comparator_datasets) < args.min_characterisation_datasets:
        console.insufficient_comparators(comparator_datasets)
        return pandda_events, autobuilds

    # Get the alignments, and save them to the object store
    time_begin_get_alignments = time.time()
    alignments: Dict[str, AlignmentInterface] = processor.process_dict(
        {_dtag: Partial(Alignment.from_structure_arrays).paramaterise(
            _dtag,
            structure_array_refs[_dtag],
            structure_array_refs[dtag],
        ) for _dtag in comparator_datasets}
    )
    alignment_refs = {_dtag: processor.put(alignments[_dtag]) for _dtag in comparator_datasets}
    time_finish_get_alignments = time.time()
    # TODO: Log properly
    # print(f"\t\tGot alignments in: {round(time_finish_get_alignments - time_begin_get_alignments, 2)}")

    # Get the reference frame and save it to the object store
    time_begin_get_frame = time.time()
    reference_frame: DFrame = DFrame(dataset, processor)
    reference_frame_ref = processor.put(reference_frame)
    time_finish_get_frame = time.time()
    # TODO: Log properly
    # print(f"\t\tGot reference frame in: {round(time_finish_get_frame - time_begin_get_frame, 2)}")

    # Get the transforms to apply to the dataset before locally aligning and save them to the object store
    transforms = [
        TruncateReflections(
            comparator_datasets,
            processing_res,
        ),
        SmoothReflections(dataset)
    ]
    transforms_ref = processor.put(transforms)

    # Load the locally aligned density maps and construct an array of them
    time_begin_get_dmaps = time.time()
    dmaps_dict = processor.process_dict(
        {
            _dtag: Partial(SparseDMapStream.parallel_load).paramaterise(
                dataset_refs[_dtag],
                alignment_refs[_dtag],
                transforms_ref,
                reference_frame_ref,
                args.debug
            )
            for _dtag
            in comparator_datasets
        }
    )
    dmaps = np.vstack([_dmap.data.reshape((1, -1)) for _dtag, _dmap in dmaps_dict.items()])
    if args.debug:
        print('Aligned dmap stats')
        for _dtag, _dmap in dmaps_dict.items():
            arr = _dmap.data
            print(f'{dtag} stats: min {np.min(arr)} max {np.max(arr)} mean {np.mean(arr)}')
    time_finish_get_dmaps = time.time()
    # TODO: log properly
    # print(f"\t\tGot dmaps in: {round(time_finish_get_dmaps - time_begin_get_dmaps, 2)}")
    dtag_array = np.array([_dtag for _dtag in comparator_datasets])

    # Get the dataset dmap, both processed and unprocessed
    dtag_index = np.argwhere(dtag_array == dtag)
    dataset_dmap_array = dmaps[dtag_index[0][0], :]
    xmap_grid = reference_frame.unmask(SparseDMap(dataset_dmap_array))
    raw_xmap_grid = dataset.reflections.transform_f_phi_to_map(sample_rate=3)
    raw_xmap_sparse = reference_frame.mask_grid(raw_xmap_grid).data
    raw_xmap_sparse_ref = processor.put(raw_xmap_sparse)
    raw_xmap_array = np.array(raw_xmap_grid, copy=True)
    raw_xmap_array_ref = processor.put(raw_xmap_array)
    # raw_xmap_grid = reference_frame.unmask(
    #     raw_xmap_sparse
    # )

    # Get the masked grid of the structure
    model_grid = get_model_map(dataset.structure.structure, xmap_grid)

    # Get the Comparator sets that define the models to try
    time_begin_get_characterization_sets = time.time()
    characterization_sets: Dict[int, Dict[str, DatasetInterface]] = get_characterization_sets(
        dtag,
        {k: v for k, v in comparator_datasets.items() if k != dtag},
        dmaps[np.array([k != dtag for k in dmaps_dict]), :],
        reference_frame,
        CharacterizationNNAndFirst(
            n_neighbours=args.min_characterisation_datasets - 1,
            min_size=args.min_characterisation_datasets - 1,
        )
    )
    time_finish_get_characterization_sets = time.time()
    # TODO: Log properly
    print(
        f"\t\tGot characterization sets in: {round(time_finish_get_characterization_sets - time_begin_get_characterization_sets, 2)}")

    # Filter the models which are clearly poor descriptions of the density
    # In theory this step could result in the exclusion of a ground state model which provided good contrast
    # for a ligand binding in one part of the protein but fit poorly to say a large disordered region
    models_to_process, model_scores, characterization_set_masks = filter_characterization_sets(
        comparator_datasets,
        characterization_sets,
        dmaps,
        dataset_dmap_array,
        reference_frame,
        PointwiseMAD(),
        process_all=False
    )
    # print(f"Models to process are {models_to_process} out of {[x for x in characterization_sets]}")

    # Plot the projections
    umap_plot_out_dir = fs.output.processed_datasets[dtag] / "model_umap"
    if not umap_plot_out_dir.exists():
        os.mkdir(umap_plot_out_dir)
    try:
        plot_aligned_density_projection(
            dmaps,
            models_to_process,
            characterization_set_masks,
            umap_plot_out_dir
        )
    except Exception as e:
        print('UMAP Failed. This is probably due to numba versioning.')
        plot_aligned_density_projection(
            dmaps,
            models_to_process,
            characterization_set_masks,
            umap_plot_out_dir,
            projection='tsne'
        )
        print(e)

    # Process the models: calculating statistical maps; using them to locate events; filtering, scoring and re-
    # filtering those events and returning those events and unpacking them
    time_begin_process_models = time.time()
    model_maps_dir = fs.output.processed_datasets[dtag] / 'model_maps'
    if not model_maps_dir.exists():
        os.mkdir(model_maps_dir)

    processed_models = {
        model_number: Partial(process_model).paramaterise(
            dataset.ligand_files,
            dataset_dmap_array,
            raw_xmap_array,
            dmaps[characterization_set_masks[model_number], :],
            reference_frame,
            reference_frame.mask_grid(model_grid).data,
            score_event,
            fs,
            model_number,
            dtag
        )()
        for model_number
        in models_to_process
    }

    model_events = {}
    model_means = {}
    model_zs = {}
    model_stds = {}
    model_metas = {}
    for model_number, result in processed_models.items():
        if result[0] is not None:
            model_events[model_number] = result[0]
        model_means[model_number] = result[1]
        model_zs[model_number] = result[2]
        # print(f'z map stats: {np.min(result[2])} {np.max(result[2])} {np.median(result[2])}')
        model_stds[model_number] = result[3]
        model_metas[model_number] = result[4]

    time_finish_process_models = time.time()
    # TODO: Log properly
    # print(f"\t\tProcessed all models in: {round(time_finish_process_models - time_begin_process_models, 2)}")

    if args.use_ligand_data & args.autobuild:
        # Build the events
        time_begin_autobuild = time.time()

        # Get the Masked processed dtag dmap array and cache
        masked_dtag_array = mask_dmap(np.copy(dataset_dmap_array), dataset.structure.structure, reference_frame)
        masked_dtag_array_ref = processor.put(masked_dtag_array)

        # Get the scoring grid for each model and cache
        masked_mean_arrays = {}
        masked_mean_array_refs = {}
        for model_number, model in model_events.items():
            masked_mean_array = mask_dmap(np.copy(model_means[model_number]), dataset.structure.structure,
                                          reference_frame)
            masked_mean_arrays[model_number] = masked_mean_array
            masked_mean_array_refs[model_number] = processor.put(masked_mean_array)

        # Cache the unmasked dtag dmap array
        unmasked_dtag_array_ref = processor.put(dataset_dmap_array)

        # Cache the unmasked mean dmap array
        unmasked_mean_array_refs = {}
        for model_number, model in model_events.items():
            unmasked_mean_array_refs[model_number] = processor.put(model_means[model_number])

        # Cache the model Z map arrays
        z_ref_arrays = {}
        for model_number, model in model_events.items():
            z_ref_arrays[model_number] = processor.put(model_zs[model_number])

        # Generate conformers of the dataset ligands to score
        conformers = {}
        conformer_refs = {}
        for ligand_key in dataset.ligand_files:
            ligand_files = dataset.ligand_files[ligand_key]
            conformers[ligand_key] = get_conformers(ligand_files)
            conformer_refs[ligand_key] = {}
            for conformer_number, conformer in conformers[ligand_key].items():
                conformer_refs[ligand_key][conformer_number] = processor.put(Structure(None, conformer))

        # Define the builds to perform
        builds_to_perform = []
        for model_number, events in model_events.items():
            for event_number, event in events.items():
                for ligand_key, ligand_conformers in conformers.items():
                    for conformer_number, conformer in ligand_conformers.items():
                        builds_to_perform.append(
                            (model_number, event_number, ligand_key, conformer_number)
                        )
        # Set up autobuilding output directory
        out_dir = fs.output.processed_datasets[dtag] / "autobuild"
        if not out_dir.exists():
            os.mkdir(out_dir)

        # Perform autobuilds of events
        # print(f"Have {len(builds_to_perform)} builds to perform!")
        builds = processor.process_dict(
            {
                _model_event_id: Partial(autobuild_conformer).paramaterise(
                    model_events[_model_event_id[0]][_model_event_id[1]].centroid,
                    model_events[_model_event_id[0]][_model_event_id[1]].bdc,
                    conformer_refs[_model_event_id[2]][_model_event_id[3]],
                    masked_dtag_array_ref,
                    masked_mean_array_refs[_model_event_id[0]],
                    reference_frame_ref,
                    out_dir,
                    f"{_model_event_id[0]}_{_model_event_id[1]}_{_model_event_id[2]}_{_model_event_id[3]}",
                    dataset_res,
                    dataset.structure,
                    unmasked_dtag_array_ref,
                    unmasked_mean_array_refs[_model_event_id[0]],
                    z_ref_arrays[_model_event_id[0]],
                    raw_xmap_sparse_ref,
                    score_build_ref,
                    raw_xmap_array_ref
                    # processing_res
                    # fs_ref,
                )
                for _model_event_id
                in builds_to_perform
            }
        )
        time_finish_autobuild = time.time()
        # TODO: Log properly
        # print(f"\t\tAutobuilt in {time_finish_autobuild - time_begin_autobuild}")

        # build_scores = score_builds(
        #     score_build,
        #     builds,
        #     raw_xmap_grid,
        #     dataset_dmap_array,
        #     model_means,
        #     model_zs
        # )

        for build_key, result in builds.items():
            for path, build in result.items():
                model_number, event_number, ligand_key, conformer_number = build_key
                # print([x for x in build.keys()])
                dmaps = {
                    'zmap': build['arr'][0][0],
                    'xmap': build['arr'][0][1],
                    'mask': build['arr'][0][2],
                }
                # for name, dmap in dmaps.items():
                #     grid = gemmi.FloatGrid(32, 32, 32)
                #     uc = gemmi.UnitCell(16.0, 16.0, 16.0, 90.0, 90.0, 90.0)
                #
                #     # uc = gemmi.UnitCell(8.0, 8.0, 8.0, 90.0, 90.0, 90.0)
                #     grid.set_unit_cell(uc)
                #
                #     grid_array = np.array(grid, copy=False)
                #     grid_array[:, :, :] = dmap[:, :, :]
                #     ccp4 = gemmi.Ccp4Map()
                #     ccp4.grid = grid
                #     ccp4.update_ccp4_header()
                #     ccp4.write_ccp4_map(str(fs.output.processed_datasets[
                #                                 dtag] / f'build_map_{model_number}_{event_number}_{ligand_key}_{conformer_number}_{name}.ccp4'))

        # Select between autobuilds and update event for each event
        for model_number, events in model_events.items():
            for event_number, event in events.items():

                event_builds = {}
                for ligand_key, ligand_conformers in conformers.items():
                    for conformer_number, conformer in ligand_conformers.items():
                        build = builds[(model_number, event_number, ligand_key, conformer_number)]
                        for build_path, result in build.items():
                            # TODO: Replace with an abstract method
                            event_builds[
                                (ligand_key, build_path, conformer_number)] = result  # + event.score #* event.score
                # TODO: Replace with an abstract method
                # selected_build_key = max(event_builds, key=lambda _key: event_builds[_key]['local_signal'])
                # selected_build_key = max(
                #     event_builds,
                #     key=lambda _key: event_builds[_key]['signal'] / event_builds[_key]['noise'],
                # )
                selected_build_key = max(
                    event_builds,
                    key=lambda _key: event_builds[_key]['score']
                )

                selected_build = event_builds[selected_build_key]

                event.build = EventBuild(
                    selected_build_key[1],
                    selected_build_key[0],
                    selected_build['score'],
                    # event_builds[selected_build_key]['signal'] / event_builds[selected_build_key]['noise'],
                    # builds[(model_number, event_number, selected_build_key[0], selected_build_key[2])][
                    #     selected_build_key[1]]['centroid'],
                    # builds[(model_number, event_number, selected_build_key[0], selected_build_key[2])][
                    #     selected_build_key[1]]['new_bdc'],
                    selected_build['centroid'],
                    selected_build['new_bdc'],
                    build_score=selected_build['score'],
                    noise=selected_build['noise'],
                    signal=selected_build['signal'],
                    num_contacts=selected_build['num_contacts'],
                    num_points=selected_build['num_points'],
                    optimal_contour=selected_build['optimal_contour'],
                    rscc=selected_build['local_signal']
                )

        # Update the event centroid and bdc from the selected build
        for model_number, events in model_events.items():
            for event_number, event in events.items():
                old_centroid = [round(float(x), 2) for x in event.centroid]
                new_centroid = [round(float(x), 2) for x in event.build.centroid]
                scores = [round(float(event.score), 2), round(float(event.build.score), 2)]
                bdcs = [round(float(event.bdc), 2), round(float(event.build.bdc), 2)]
                # print(
                #     f"{model_number} : {event_number} : {old_centroid} : {new_centroid} : {scores} : {bdcs} : {Path(event.build.build_path).name}")
                event.centroid = event.build.centroid
                event.bdc = event.build.bdc

        # Seperate updated model events by model number
        update_model_events = {}
        for model_number, events in model_events.items():
            for event_number, event in events.items():
                if model_number not in update_model_events:
                    update_model_events[model_number] = {}
                update_model_events[model_number][event_number] = event

        # Filter events by builds
        for model_number in update_model_events:
            for filter in [
                FilterSymmetryPosBuilds(dataset, 2.0),
                FilterLocallyHighestBuildScoring(10.0)
            ]:
                j_0 = len(update_model_events[model_number])
                update_model_events[model_number] = filter(update_model_events[model_number])
                # TODO: Log properly
                # print(
                #     f"\t\t\tModel {model_number} when from {j_0} to {len(update_model_events[model_number])} events of filter {filter}")


    else:  # args.use_ligand_data==False
        # Add dummy builds
        for model_number, events in model_events.items():
            for event_number, event in events.items():
                event.build = EventBuild(
                    'None',
                    'None',
                    event.score,
                    event.centroid,
                    event.bdc,
                    build_score=event.score,
                    noise=0,
                    signal=0,
                    num_contacts=0,
                    num_points=0,
                    optimal_contour=0,
                    rscc=0
                )
        # Seperate updated model events by model number
        update_model_events = {}
        for model_number, events in model_events.items():
            for event_number, event in events.items():
                if model_number not in update_model_events:
                    update_model_events[model_number] = {}
                update_model_events[model_number][event_number] = event

    # Filter models by whether they have events and skip if no models remain
    model_events = {model_number: events for model_number, events in update_model_events.items() if
                    len(events) > 0}
    if len(model_events) == 0:
        # TODO: Log properly
        print(f"NO EVENTS FOR DATASET {dtag}: SKIPPING REST OF PROCESSING!")
        selected_model_num = models_to_process[0]
        selected_events = {}
        top_selected_model_events = {}

    else:
        # Select a model based on the events it produced and get the associated events
        selected_model_num, selected_events = select_model(model_events)

        # Filter the events to select those to output event maps for and to autobuild
        # This step can be dangerous in that events with high multiplity (for example due to NCS) could be filtered
        top_selected_model_events = filter_selected_events(dtag, selected_events, )

    for event_id, event in top_selected_model_events.items():
        pandda_events[event_id] = event

    for event_id, event in top_selected_model_events.items():
        autobuilds[event_id] = {
            event.build.ligand_key: AutobuildResult(
                {event.build.build_path: {'score': event.build.signal,
                                          'centroid': event.build.centroid}},
                None, None, None, None, None
            )
        }
    # Output event maps and model maps
    time_begin_output_maps = time.time()
    # print(
    #     f'z map stats: {np.min(model_zs[selected_model_num])} {np.max(model_zs[selected_model_num])} {np.median(model_zs[selected_model_num])} {np.sum(np.isnan(model_zs[selected_model_num]))}')

    # for event in top_selected_model_events.values():
    #     print(f'{event.bdc} : {event.build.bdc}')
    output_maps(
        dtag,
        fs,
        top_selected_model_events,
        dataset_dmap_array,
        model_means[selected_model_num],
        model_zs[selected_model_num],
        reference_frame,
        processing_res,
        model_events,
        model_means,
        model_stds,
        model_zs
    )

    # Canonicalize paths to best autobuild for each event
    if args.use_ligand_data & args.autobuild:
        for event_id, event in top_selected_model_events.items():
            shutil.copy(
                event.build.build_path,
                fs.output.processed_datasets[dtag] / f'{dtag}_event_{str(event_id[1])}_best_autobuild.pdb'
            )

    time_finish_output_maps = time.time()
    # TODO: Log properly
    # print(f"\t\tOutput maps in: {round(time_finish_output_maps - time_begin_output_maps, 2)}")

    time_finish_process_dataset = time.time()
    # TODO: Log properly
    # print(f"\tProcessed dataset in {round(time_finish_process_dataset - time_begin_process_dataset, 2)}")

    # Serialize information on dataset processing to a human readable yaml file
    serialize.processed_dataset(
        comparator_datasets,
        processing_res,
        characterization_sets,
        model_scores,
        models_to_process,
        processed_models,
        selected_model_num,
        top_selected_model_events,
        reference_frame,
        model_metas,
        fs.output.processed_datasets[dtag] / f"processed_dataset.yaml"
    )
    serialize.serialize_events(
        {event_id[1]: event for event_id, event in top_selected_model_events.items()},
        fs.output.processed_datasets[dtag] / f"events.yaml"
    )

    console.processed_dataset(
        dtag,
        dataset,
        comparator_datasets,
        processing_res,
        characterization_sets,
        models_to_process,
        processed_models,
        selected_model_num,
        top_selected_model_events,
    )

    return pandda_events, autobuilds
