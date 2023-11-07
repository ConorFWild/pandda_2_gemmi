import os
import time

from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import gemmi

from pandda_gemmi.interfaces import *

from pandda_gemmi import serialize

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
    FilterCompatibleStructures
)
from pandda_gemmi.event_model.event import EventBuild
from pandda_gemmi.event_model.characterization import get_characterization_sets, CharacterizationNN, CharacterizationNNAndFirst
from pandda_gemmi.event_model.filter_characterization_sets import filter_characterization_sets
from pandda_gemmi.event_model.outlier import PointwiseNormal
from pandda_gemmi.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.event_model.score import get_model_map, ScoreCNNLigand
from pandda_gemmi.event_model.filter import FilterSize, FilterScore, FilterLocallyHighestLargest, FilterLocallyHighestScoring, FilterSymmetryPosBuilds, FilterLocallyHighestBuildScoring
from pandda_gemmi.event_model.select import select_model
from pandda_gemmi.event_model.output import output_maps
from pandda_gemmi.event_model.filter_selected_events import filter_selected_events

from pandda_gemmi.site_model import HeirarchicalSiteModel, Site, get_sites

from pandda_gemmi.autobuild import autobuild, autobuild_model_event, AutobuildResult
from pandda_gemmi.autobuild.inbuilt import AutobuildInbuilt, AutobuildModelEventInbuilt, mask_dmap, get_conformers, autobuild_conformer
from pandda_gemmi.autobuild.merge import merge_autobuilds, MergeHighestBuildAndEventScore
from pandda_gemmi.autobuild.preprocess_structure import AutobuildPreprocessStructure
from pandda_gemmi.autobuild.preprocess_dmap import AutobuildPreprocessDMap

from pandda_gemmi.ranking import rank_events, RankHighEventScore, RankHighBuildScore, RankHighEventBuildScore

from pandda_gemmi.tables import output_tables
from pandda_gemmi.pandda_logging import PanDDAConsole

class GetDatasetsToProcess:
    def __init__(self, filters=[]):
        self.filters = filters

    def __call__(self,
                 #*args, **kwargs
                 datasets: Dict[str, DatasetInterface],
                 fs: PanDDAFSInterface
                 ):
        # datasets_to_process = {}
        datasets_not_to_process = {}

        remaining_datasets = {_dtag: _dataset for _dtag, _dataset in datasets.items()}
        for _filter in self.filters:
            remaining_datasets = _filter(remaining_datasets)
            for dtag in datasets:
                if (dtag not in datasets_not_to_process) and (dtag not in remaining_datasets):
                    datasets_not_to_process[dtag] = _filter.description()

        return {_k: remaining_datasets[_k] for _k in sorted(remaining_datasets)}, {_k: datasets_not_to_process[_k] for _k in sorted(datasets_not_to_process)}

class ProcessModel:
    def __init__(self,
                 minimum_z_cluster_size=5.0,
                 minimum_event_score=0.17,
                 local_highest_score_radius=8.0
                 ):
        self.minimum_z_cluster_size = minimum_z_cluster_size
        self.minimum_event_score = minimum_event_score
        self.local_highest_score_radius = local_highest_score_radius


    def __call__(self, #*args, **kwargs
        ligand_files,
        homogenized_dataset_dmap_array,
        dataset_dmap_array,
        characterization_set_dmaps_array,
        reference_frame,
        model_map,
        score,
):
        # Get the statical maps
        mean, std, z = PointwiseNormal()(
            homogenized_dataset_dmap_array,
            characterization_set_dmaps_array
        )

        mean_grid = reference_frame.unmask(SparseDMap(mean))
        z_grid = reference_frame.unmask(SparseDMap(z))

        xmap_grid = reference_frame.unmask(SparseDMap(homogenized_dataset_dmap_array))

        raw_xmap_grid = gemmi.FloatGrid(*dataset_dmap_array.shape)
        raw_xmap_grid.set_unit_cell(z_grid.unit_cell)
        raw_xmap_grid_array = np.array(raw_xmap_grid, copy=False)
        raw_xmap_grid_array[:, :, :] = dataset_dmap_array[:, :, :]

        median = np.median(mean[reference_frame.mask.indicies_sparse_inner_atomic])

        model_grid = reference_frame.unmask(SparseDMap(model_map))

        # Get the initial events from clustering the Z map
        events = ClusterDensityDBSCAN()(z, reference_frame)

        if len(events) == 0:
            return None, None, None

        # Filter the events prior to scoring them based on their size
        for filter in [
            FilterSize(reference_frame, min_size=self.minimum_z_cluster_size),
        ]:
            events = filter(events)

        # Return None if there are no events after pre-scoring filters
        if len(events) == 0:
            return None, None, None

        # Score the events with some method such as the CNN
        time_begin_score_events = time.time()
        events = score(ligand_files, events, xmap_grid, raw_xmap_grid, mean_grid, z_grid, model_grid,
                       median, reference_frame, homogenized_dataset_dmap_array, mean
                       )
        time_finish_score_events = time.time()

        # Filter the events after scoring based on their score and keeping only the locally highest scoring event
        num_events = len(events)
        for filter in [
            FilterScore(self.minimum_event_score),  # Filter events based on their score
            # FilterLocallyHighestLargest(self.local_highest_score_radius),  # Filter events that are close to other,
            #                                                                # better scoring events
            # FilterLocallyHighestScoring(self.local_highest_score_radius)
        ]:
            events = filter(events)

        print(f"Filtered {num_events} down to {len(events)}")


        # # Build the events
        #
        # ...
        # # Update centroid from build
        #
        # # Filter events by builds
        # for filter in [
        #     # FilterScore(self.minimum_event_score),  # Filter events based on their score
        #     # FilterLocallyHighestLargest(self.local_highest_score_radius),  # Filter events that are close to other,
        #     #                                                                # better scoring events
        #     FilterLocallyHighestBuildScoring(self.local_highest_score_radius)
        # ]:
        #     events = filter(events)


        # Return None if there are no events after post-scoring filters
        if len(events) == 0:
            return None, None, None

        events = {j + 1: event for j, event in enumerate(events.values())}

        return events, mean, z


def pandda(args: PanDDAArgs):
    # Record time at which PanDDA processing begins
    time_pandda_begin = time.time()

    # Create the console to print output throughout the programs run
    console = PanDDAConsole()

    # Print the PanDDA initialization message and the command line arguments
    console.start_pandda()
    console.start_parse_command_line_args()
    console.summarise_arguments(args)

    # Get the processor to handle the dispatch of functions to multiple cores and the cache of parallel
    # processed objects
    console.start_initialise_multiprocessor()
    processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)
    console.print_initialized_local_processor(args)

    # Get the model of the input and output of the program on the file systems
    console.start_fs_model()
    fs: PanDDAFSInterface = PanDDAFS(Path(args.data_dirs), Path(args.out_dir), args.pdb_regex, args.mtz_regex)
    console.summarise_fs_model(fs)
    fs_ref = processor.put(fs)

    # Get the method for scoring events
    score = ScoreCNNLigand()

    # Get the method for processing the statistical models
    process_model = ProcessModel()

    # Load the structures and reflections from the datasets found in the file system, and create references to these
    # dataset objects and the arrays of their structures in the multiprocessing cache
    console.start_load_datasets()
    datasets: Dict[str, DatasetInterface] = {
        dataset_dir.dtag: XRayDataset.from_paths(
            dataset_dir.input_pdb_file,
            dataset_dir.input_mtz_file,
            dataset_dir.input_ligands,
        )
        for dataset_dir
        in fs.input.dataset_dirs.values()
    }
    dataset_refs = {_dtag: processor.put(datasets[_dtag]) for _dtag in datasets}
    structure_array_refs = {_dtag: processor.put(StructureArray.from_structure(datasets[_dtag].structure)) for _dtag in
                            datasets}

    # Summarise the datasets loaded from the file system and serialize the information on the input into a human
    # readable yaml file
    console.summarise_datasets(datasets, fs)
    serialize.input_data(
        fs, datasets, fs.output.path / "input.yaml"
    )

    # Get the datasets to process
    dataset_to_process, datasets_not_to_process = GetDatasetsToProcess(
        [
            FilterRFree(args.max_rfree),
            FilterRange(args.dataset_range),
            FilterExcludeFromAnalysis(args.exclude_from_z_map_analysis),
            FilterOnlyDatasets(args.only_datasets)
        ]
    )(datasets, fs)
    console.summarize_datasets_to_process(dataset_to_process, datasets_not_to_process)

    # Process each dataset by identifying potential comparator datasets, constructing proposed statistical models,
    # calculating alignments of comparator datasets, locally aligning electron density, filtering statistical models
    # to the plausible set, evaluating those models for events, selecting a model to take forward based on those events
    # and outputing event maps, z maps and mean maps for that model
    pandda_events = {}
    time_begin_process_datasets = time.time()
    console.start_process_shells()
    autobuilds = {}
    for j, dtag in enumerate(dataset_to_process):

        # Print basic information of the dataset to be processed

        # Record the time that dataset processing begins
        time_begin_process_dataset = time.time()

        events_yaml_path = fs.output.processed_datasets[dtag] / f"events.yaml"
        print(f"Checking for a event yaml at: {events_yaml_path}")
        if events_yaml_path.exists():
            print(f"Already have events for dataset! Skipping!")
            _events = serialize.unserialize_events(fs.output.processed_datasets[dtag] / f"events.yaml")
            for event_idx, event in _events.items():
                pandda_events[(dtag, event_idx)] = event
            # print(pandda_events)
            for event_idx, event in _events.items():
                autobuilds[(dtag, event_idx)] = {
                    event.build.ligand_key: AutobuildResult(
                        {event.build.build_path: {'score': event.build.score, 'centroid': event.build.centroid}},
                        None, None, None, None, None
                    )
                }
            continue

        # Get the dataset
        dataset = datasets[dtag]

        # Skip processing the dataset if there is no ligand data
        if len([_key for _key in dataset.ligand_files if dataset.ligand_files[_key].ligand_cif]) == 0:
            console.no_ligand_data()
            continue

        # Get the resolution of the dataset
        dataset_res = dataset.reflections.resolution()

        # Get the comparator datasets: these are filtered for reasonable data quality, space group compatability,
        # compatability of structural models and similar resolution
        comparator_datasets: Dict[str, DatasetInterface] = get_comparators(
            datasets,
            [
                FilterRFree(args.max_rfree),
                FilterSpaceGroup(dataset),
                FilterCompatibleStructures(dataset),
                FilterResolution(dataset_res, args.max_shell_datasets, 100, args.high_res_buffer)]
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
            dataset_to_process,
            time_begin_process_datasets
        )

        # Skip if there are insufficient comparators in order to characterize a statistical model
        if len(comparator_datasets) < args.min_characterisation_datasets:
            console.insufficient_comparators(comparator_datasets)
            continue

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
        print(f"\t\tGot alignments in: {round(time_finish_get_alignments - time_begin_get_alignments, 2)}")

        # Get the reference frame and save it to the object store
        time_begin_get_frame = time.time()
        reference_frame: DFrame = DFrame(dataset, processor)
        reference_frame_ref = processor.put(reference_frame)
        time_finish_get_frame = time.time()
        print(f"\t\tGot reference frame in: {round(time_finish_get_frame - time_begin_get_frame, 2)}")

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
                    reference_frame_ref
                )
                for _dtag
                in comparator_datasets
            }
        )
        dmaps = np.vstack([_dmap.data.reshape((1, -1)) for _dtag, _dmap in dmaps_dict.items()])
        time_finish_get_dmaps = time.time()
        print(f"\t\tGot dmaps in: {round(time_finish_get_dmaps - time_begin_get_dmaps, 2)}")
        dtag_array = np.array([_dtag for _dtag in comparator_datasets])

        # Get the dataset dmap
        dtag_index = np.argwhere(dtag_array == dtag)
        # print(f"Dtag index: {dtag_index}")
        dataset_dmap_array = dmaps[dtag_index[0][0], :]
        xmap_grid = reference_frame.unmask(SparseDMap(dataset_dmap_array))

        raw_xmap_grid = dataset.reflections.transform_f_phi_to_map(sample_rate=3)
        raw_xmap_array = np.array(raw_xmap_grid, copy=True)

        # Get the masked grid of the structure
        model_grid = get_model_map(dataset.structure.structure, xmap_grid)

        # Get the Comparator sets that define the models to try
        time_begin_get_characterization_sets = time.time()
        characterization_sets: Dict[int, Dict[str, DatasetInterface]] = get_characterization_sets(
            dtag,
            comparator_datasets,
            dmaps,
            reference_frame,
            # CharacterizationGaussianMixture(
            #     n_components=min(20, int(len(comparator_datasets) / 25)),
            #     covariance_type="full",
            # ),
            CharacterizationNNAndFirst()
        )
        time_finish_get_characterization_sets = time.time()
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
            PointwiseNormal(),
        )

        # print(f"Models to process: {models_to_process}")
        # serialize.models()

        # Process the models: calculating statistical maps; using them to locate events; filtering, scoring and re-
        # filtering those events and returning those events and unpacking them
        time_begin_process_models = time.time()

        processed_models = {
            model_number: Partial(process_model).paramaterise(
                dataset.ligand_files,
                dataset_dmap_array,
                raw_xmap_array,
                dmaps[characterization_set_masks[model_number], :],
                reference_frame,
                reference_frame.mask_grid(model_grid).data,
                score
            )()
            for model_number
            in models_to_process
        }

        model_events = {}
        model_means = {}
        model_zs = {}
        for model_number, result in processed_models.items():
            if result[0] is not None:
                model_events[model_number] = result[0]
                model_means[model_number] = result[1]
                model_zs[model_number] = result[2]

        time_finish_process_models = time.time()
        print(f"\t\tProcessed all models in: {round(time_finish_process_models - time_begin_process_models, 2)}")

        # Build the events
        time_begin_autobuild = time.time()

        # Masked dtag array
        masked_dtag_array = mask_dmap(np.copy(dataset_dmap_array), dataset.structure.structure, reference_frame)
        masked_dtag_array_ref = processor.put(masked_dtag_array)

        # Get the scoring grid
        masked_mean_arrays = {}
        masked_mean_array_refs = {}
        for model_number, model in model_events.items():
            # for event_number, event in events.items():
            masked_mean_array = mask_dmap(np.copy(model_means[model_number]), dataset.structure.structure, reference_frame)
            masked_mean_arrays[model_number] = masked_mean_array
            masked_mean_array_refs[model_number] = processor.put(masked_mean_array)

        # Generate conformers to score
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

        print(builds_to_perform)

        out_dir = fs.output.processed_datasets[dtag] / "autobuild"
        if not out_dir.exists():
            os.mkdir(out_dir)
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
                    dataset_res
                    # processing_res
                    # fs_ref,
                )
                for _model_event_id
                in builds_to_perform
            }
        )
        time_finish_autobuild = time.time()
        print(f"\t\tAutobuilt in {time_finish_autobuild - time_begin_autobuild}")
        # raise Exception

        # events_to_process = {}
        # for model_number, model_events in model_events.items():
        #     for event_number, event in model_events.items():
        #         events_to_process[(model_number, event_number)] = event
        #
        # event_autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]] = processor.process_dict(
        #     {
        #         _model_event_id: Partial(autobuild_model_event).paramaterise(
        #             dtag,
        #             _model_event_id,
        #             dataset_refs[dtag],
        #             events_to_process[_model_event_id],
        #             dataset_dmap_array,
        #             reference_frame_ref,
        #             AutobuildPreprocessStructure(),
        #             AutobuildPreprocessDMap(),
        #             # Rhofit(cut=1.0),
        #             AutobuildModelEventInbuilt(),
        #             fs_ref
        #         )
        #         for _model_event_id
        #         in events_to_process
        #     }
        # )
        #
        # # Select between autobuilds and update event for each event
        # for event_id, autobuild_results in event_autobuilds.items():
        #     event = events_to_process[event_id]
        #     builds = {}
        #     for ligand_key, autobuild_result in autobuild_results.items():
        #         for build_path, result in autobuild_result.log_result_dict.items():
        #             builds[(ligand_key, build_path)] = result['score']
        #
        #     selected_build_key = max(builds, key=lambda _key: -builds[_key])
        #
        #     event.build = EventBuild(
        #         selected_build_key[1],
        #         selected_build_key[0],
        #         builds[selected_build_key],
        #         event_autobuilds[event_id][selected_build_key[0]].log_result_dict[selected_build_key[1]]['centroid']
        #     )

        # Select between autobuilds and update event for each event
        for model_number, events in model_events.items():
            # event = events_to_process[event_id]
            for event_number, event in events.items():

                event_builds = {}
                for ligand_key, ligand_conformers in conformers.items():
                    for conformer_number, conformer in ligand_conformers.items():
                        build = builds[(model_number, event_number, ligand_key, conformer_number)]
                        for build_path, result in build.items():
                            # event_builds[(ligand_key, build_path, conformer_number)] = result['score']
                            event_builds[(ligand_key, build_path, conformer_number)] = result['local_signal'] #+ event.score #* event.score

                selected_build_key = max(event_builds, key=lambda _key: event_builds[_key])

                event.build = EventBuild(
                    selected_build_key[1],
                    selected_build_key[0],
                    event_builds[selected_build_key],
                    builds[(model_number,event_number,selected_build_key[0],selected_build_key[2])][selected_build_key[1]]['centroid'],
                    builds[(model_number, event_number, selected_build_key[0], selected_build_key[2])][selected_build_key[1]]['new_bdc'],
                )
                # event.build = EventBuild(
                #     None,
                #     list(conformers.keys())[0],
                #     event.score,
                #     event.centroid
                # )

        # Update centroid from build
        # for event_id, event in events_to_process.items():
        for model_number, events in model_events.items():
            for event_number, event in events.items():
                old_centroid = [round(float(x), 2) for x in event.centroid]
                new_centroid = [round(float(x), 2) for x in event.build.centroid]
                scores = [round(float(event.score), 2), round(float(event.build.score), 2)]
                bdcs = [round(float(event.bdc), 2), round(float(event.build.bdc), 2)]
                print(f"{model_number} : {event_number} : {old_centroid} : {new_centroid} : {scores} : {bdcs} : {Path(event.build.build_path).name}")
                event.centroid = event.build.centroid
                event.score = event.build.score
                event.bdc = event.build.bdc

        # Seperate by model number
        update_model_events = {}
        # for (model_number, event_number), event in events_to_process.items():
        for model_number, events in model_events.items():
            for event_number, event in events.items():
                if model_number not in update_model_events:
                    update_model_events[model_number] = {}
                update_model_events[model_number][event_number] = event
        print(f"Updated Model Events")
        print(update_model_events)


        # Filter events by builds
        for model_number in update_model_events:
            for filter in [
                FilterSymmetryPosBuilds(dataset, 2.0),
                FilterLocallyHighestBuildScoring(10.0)
            ]:
                j_0 = len(update_model_events[model_number])
                update_model_events[model_number] = filter(update_model_events[model_number])
                print(f"\t\t\tModel {model_number} when from {j_0} to {len(update_model_events[model_number])} events of local filter")

        # time_finish_autobuild = time.time()
        # print(f"\t\tAutobuilt in {time_finish_autobuild-time_begin_autobuild}")


        model_events = {model_number: events for model_number, events in update_model_events.items() if len(events) > 0}
        if len(model_events) == 0:
            print(f"NO EVENTS FOR DATASET {dtag}: SKIPPING REST OF PROCESSING!")
            continue

        # Select a model based on the events it produced and get the associated events
        selected_model_num, selected_events = select_model(model_events)
        # print(f"Selected model number: {selected_model_num}")

        # Filter the events to select those to output event maps for and to autobuild
        # This step can be dangerous in that events with high multiplity (for example due to NCS) could be filtered
        top_selected_model_events = filter_selected_events(dtag, selected_events)

        for event_id, event in top_selected_model_events.items():
            pandda_events[event_id] = event

        for event_id, event in top_selected_model_events.items():
            autobuilds[event_id] = {
                    event.build.ligand_key: AutobuildResult(
                        {event.build.build_path: {'score': event.build.score, 'centroid': event.build.centroid}},
                        None, None, None, None, None
                    )
                }

        # Output event maps and model maps
        time_begin_output_maps = time.time()
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
            model_means
        )
        time_finish_output_maps = time.time()
        print(f"\t\tOutput maps in: {round(time_finish_output_maps - time_begin_output_maps, 2)}")

        time_finish_process_dataset = time.time()
        print(f"\tProcessed dataset in {round(time_finish_process_dataset - time_begin_process_dataset, 2)}")

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

    time_finish_process_datasets = time.time()
    print(
        f"Processed {len(datasets)} datasets in {round(time_finish_process_datasets - time_begin_process_datasets, 2)}")

    # Autobuild the best scoring event for each dataset
    console.start_autobuilding()

    # Record the time at which autobuilding begins
    time_begin_autobuild = time.time()

    # Get the events to autobuild
    # best_events = {}
    # for dtag in datasets:
    #     dtag_events = {_event_id: pandda_events[_event_id] for _event_id in pandda_events if _event_id[0] == dtag}
    #     if len(dtag_events) == 0:
    #         continue
    #     best_dtag_event_id = max(dtag_events, key=lambda _event_id: dtag_events[_event_id].score)
    #     best_events[best_dtag_event_id] = pandda_events[best_dtag_event_id]

    # processor.shutdown()
    # processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)
    # dataset_refs = {_dtag: processor.put(datasets[_dtag]) for _dtag in datasets}
    # fs_ref = processor.put(fs)

    time_autobuild_begin = time.time()
    autobuild_yaml_path = fs.output.path / "autobuilds.yaml"
    if autobuild_yaml_path.exists():
        autobuilds = serialize.unserialize_autobuilds(autobuild_yaml_path)
    #
    else:
    #     event_autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]] = processor.process_dict(
    #         {
    #             _event_id: Partial(autobuild).paramaterise(
    #                 _event_id,
    #                 dataset_refs[_event_id[0]],
    #                 pandda_events[_event_id],
    #                 AutobuildPreprocessStructure(),
    #                 AutobuildPreprocessDMap(),
    #                 # Rhofit(cut=1.0),
    #                 AutobuildInbuilt(),
    #                 fs_ref
    #             )
    #             for _event_id
    #             in pandda_events
    #         }
    #     )

        time_autobuild_finish = time.time()
        # print(f"Autobuilt in: {time_autobuild_finish - time_autobuild_begin}")

        # autobuilds = {}
        # for _event_id in pandda_events:
        #     if _event_id in event_autobuilds:
        #         autobuilds[_event_id] = event_autobuilds[_event_id]
        #     else:
        #         autobuilds[_event_id] = {ligand_key: AutobuildResult(None, None, None, None, None, None) for ligand_key in
        #                                  datasets[_event_id[0]].ligand_files}
        time_finish_autobuild = time.time()
        # print(f"Autobuilt {len(pandda_events)} events in: {round(time_finish_autobuild - time_begin_autobuild, 1)}")
        # console.summarise_autobuilding(autobuild_results)

        # Merge the autobuilds into PanDDA output models
        merged_build_scores = merge_autobuilds(
            datasets,
            pandda_events,
            autobuilds,
            fs,
            # MergeHighestRSCC(),
            # MergeHighestBuildScore()
            MergeHighestBuildAndEventScore()
        )

        #
        console.processed_autobuilds(autobuilds)

        #
        # serialize.processed_autobuilds(
        #     datasets,
        #     event_autobuilds,
        #     fs.output.path / "autobuilds.yaml"
        # )

    # Get the sites
    structure_array_refs = {_dtag: processor.put(StructureArray.from_structure(datasets[_dtag].structure)) for _dtag in
                            datasets}
    sites: Dict[int, Site] = get_sites(
        datasets,
        pandda_events,
        processor,
        structure_array_refs,
        HeirarchicalSiteModel(t=args.max_site_distance_cutoff)
    )
    print("Sites")
    for site_id, site in sites.items():
        print(f"{site_id} : {site.centroid} : {site.event_ids}")

    # Rank the events for display in PanDDA inspect
    ranking = rank_events(
        pandda_events,
        autobuilds,
        RankHighEventScore(),
        # RankHighBuildScore()
        # RankHighEventBuildScore()
    )
    for event_id in ranking:
        print(f"{event_id} : {round(pandda_events[event_id].build.score, 2)}")

    # Output the event and site tables
    output_tables(pandda_events, ranking, sites, fs)
    time_pandda_finish = time.time()
    print(f"PanDDA ran in: {round(time_pandda_finish - time_pandda_begin, 2)} seconds!")