import time

import numpy as np
import gemmi

from pandda_gemmi.interfaces import *

from pandda_gemmi import serialize

from pandda_gemmi.args import PanDDAArgs

from pandda_gemmi.fs import PanDDAFS
from pandda_gemmi.dataset import XRayDataset, StructureArray
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
    FilterSpaceGroup,
    FilterResolution,
    FilterCompatibleStructures
)
from pandda_gemmi.event_model.characterization import get_characterization_sets, CharacterizationNN
from pandda_gemmi.event_model.filter_characterization_sets import filter_characterization_sets
from pandda_gemmi.event_model.outlier import PointwiseNormal
from pandda_gemmi.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.event_model.score import get_model_map, ScoreCNNLigand
from pandda_gemmi.event_model.filter import FilterSize, FilterScore, FilterLocallyHighestLargest
from pandda_gemmi.event_model.select import select_model
from pandda_gemmi.event_model.output import output_maps
from pandda_gemmi.event_model.filter_selected_events import filter_selected_events

from pandda_gemmi.site_model import HeirarchicalSiteModel, Site, get_sites

from pandda_gemmi.autobuild import autobuild, AutobuildResult
from pandda_gemmi.autobuild.inbuilt import AutobuildInbuilt
from pandda_gemmi.autobuild.merge import merge_autobuilds, MergeHighestBuildAndEventScore
from pandda_gemmi.autobuild.preprocess_structure import AutobuildPreprocessStructure
from pandda_gemmi.autobuild.preprocess_dmap import AutobuildPreprocessDMap

from pandda_gemmi.ranking import rank_events, RankHighEventScore

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

        return remaining_datasets, datasets_not_to_process

class ProcessModel:
    def __init__(self,
                 minimum_z_cluster_size=5.0,
                 minimum_event_score=0.17,
                 local_highest_score_radius=5.0
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
                       median,
                       )
        time_finish_score_events = time.time()

        # Filter the events after scoring based on their score and keeping only the locally highest scoring event
        for filter in [
            FilterScore(self.minimum_event_score),  # Filter events based on their score
            FilterLocallyHighestLargest(self.local_highest_score_radius),  # Filter events that are close to other,
                                                                           # better scoring events
        ]:
            events = filter(events)

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
    fs: PanDDAFSInterface = PanDDAFS(Path(args.data_dirs), Path(args.out_dir))
    console.summarise_fs_model(fs)

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
    dataset_to_process, datasets_not_to_process = GetDatasetsToProcess([FilterRFree(args.max_rfree),])(datasets, fs)
    console.summarize_datasets_to_process(dataset_to_process, datasets_not_to_process)

    # Process each dataset by identifying potential comparator datasets, constructing proposed statistical models,
    # calculating alignments of comparator datasets, locally aligning electron density, filtering statistical models
    # to the plausible set, evaluating those models for events, selecting a model to take forward based on those events
    # and outputing event maps, z maps and mean maps for that model
    pandda_events = {}
    time_begin_process_datasets = time.time()
    console.start_process_shells()
    for dtag in datasets:
        # Print basic information of the dataset to be processed

        # Record the time that dataset processing begins
        time_begin_process_dataset = time.time()

        if (fs.output.processed_datasets[dtag] / f"events.yaml").exists():
            pandda_events[dtag] = {
                (dtag, event_idx): event
                for event_idx, event
                in serialize.unserialize_events(fs.output.processed_datasets[dtag] / f"events.yaml").items()
            }
            continue

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
                FilterCompatibleStructures(dataset),
                FilterResolution(dataset_res, args.max_shell_datasets, args.high_res_buffer)]
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
            processing_res
        )

        # Skip if there are insufficient comparators in order to characterize a statistical model
        if len(comparator_datasets) < args.min_characterisation_datasets:
            console.insufficient_comparators(comparator_datasets)
            continue

        # Skip processing the dataset if there is no ligand data
        if len(dataset.ligand_files) == 0:
            console.no_ligand_data()
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
            CharacterizationNN()
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
        model_events = {model_number: events for model_number, events in model_events.items() if len(events) > 0}
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
            processing_res
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

    processor.shutdown()
    processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)
    dataset_refs = {_dtag: processor.put(datasets[_dtag]) for _dtag in datasets}
    fs_ref = processor.put(fs)

    time_autobuild_begin = time.time()

    event_autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]] = processor.process_dict(
        {
            _event_id: Partial(autobuild).paramaterise(
                _event_id,
                dataset_refs[_event_id[0]],
                pandda_events[_event_id],
                AutobuildPreprocessStructure(),
                AutobuildPreprocessDMap(),
                # Rhofit(cut=1.0),
                AutobuildInbuilt(),
                fs_ref
            )
            for _event_id
            in pandda_events
        }
    )

    time_autobuild_finish = time.time()
    # print(f"Autobuilt in: {time_autobuild_finish - time_autobuild_begin}")

    autobuilds = {}
    for _event_id in pandda_events:
        if _event_id in event_autobuilds:
            autobuilds[_event_id] = event_autobuilds[_event_id]
        else:
            autobuilds[_event_id] = {ligand_key: AutobuildResult(None, None, None, None, None, None) for ligand_key in
                                     datasets[_event_id[0]].ligand_files}
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
    serialize.processed_autobuilds(
        datasets,
        event_autobuilds,
        fs.output.path / "autobuilds.yaml"
    )

    # Get the sites
    sites: Dict[int, Site] = get_sites(
        pandda_events,
        HeirarchicalSiteModel(t=args.max_site_distance_cutoff)
    )
    # for site_id, site in sites.items():
    #     print(f"{site_id} : {site.centroid} : {site.event_ids}")

    # Rank the events for display in PanDDA inspect
    ranking = rank_events(
        pandda_events,
        autobuilds,
        RankHighEventScore(),
        # RankHighBuildScore()
    )
    # for event_id in ranking:
    #     print(f"{event_id} : {round(pandda_events[event_id].score, 2)}")

    # Output the event and site tables
    output_tables(pandda_events, ranking, sites, fs)
    time_pandda_finish = time.time()
    print(f"PanDDA ran in: {time_pandda_finish - time_pandda_begin}")


if __name__ == '__main__':
    # Parse Command Line Arguments
    args = PanDDAArgs.from_command_line()

    # Process the PanDDA
    pandda(args)
