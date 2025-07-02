import os
import shutil
import time
import inspect

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    print('No sklearn-express available!')

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


class GetDatasetsToProcess:
    def __init__(self, filters=None):
        self.filters = filters

    def __call__(self,
                 # *args, **kwargs
                 datasets: Dict[str, DatasetInterface],
                 fs: PanDDAFSInterface
                 ):
        datasets_not_to_process = {}
        remaining_datasets = {_dtag: _dataset for _dtag, _dataset in datasets.items()}
        for _filter in self.filters:
            remaining_datasets = _filter(remaining_datasets)
            for dtag in datasets:
                if (dtag not in datasets_not_to_process) and (dtag not in remaining_datasets):
                    datasets_not_to_process[dtag] = _filter.description()

        sorted_remaining_datasets = {
            _k: remaining_datasets[_k]
            for _k
            in sorted(remaining_datasets)
        }
        sorted_datasets_not_to_process = {
            _k: datasets_not_to_process[_k]
            for _k
            in sorted(datasets_not_to_process)
        }
        return sorted_remaining_datasets, sorted_datasets_not_to_process


class ProcessModel:
    def __init__(self,
                 minimum_z_cluster_size=5.0,
                 minimum_event_score=0.15,
                 local_highest_score_radius=8.0,
                 use_ligand_data=True,
                 debug=False
                 ):
        self.minimum_z_cluster_size = minimum_z_cluster_size
        self.minimum_event_score = minimum_event_score
        self.local_highest_score_radius = local_highest_score_radius
        self.use_ligand_data = use_ligand_data
        self.debug=debug

    def __call__(self,  # *args, **kwargs
                 ligand_files,
                 homogenized_dataset_dmap_array,
                 dataset_dmap_array,
                 characterization_set_dmaps_array,
                 reference_frame,
                 model_map,
                 score,
                 fs,
                 model_number,
                 dtag
                 ):
        # Get the statical maps
        mean, std, z = PointwiseMAD()(
            homogenized_dataset_dmap_array,
            characterization_set_dmaps_array
        )

        mean_grid = reference_frame.unmask(SparseDMap(mean))
        z_grid = reference_frame.unmask(SparseDMap((z - np.mean(z)) / np.std(z)))

        # Get the median
        protein_density_median = np.median(mean[reference_frame.mask.indicies_sparse_inner_atomic])

        # unsparsify input maps
        xmap_grid = reference_frame.unmask(SparseDMap(homogenized_dataset_dmap_array))
        raw_xmap_grid = gemmi.FloatGrid(*dataset_dmap_array.shape)
        raw_xmap_grid.set_unit_cell(z_grid.unit_cell)
        raw_xmap_grid_array = np.array(raw_xmap_grid, copy=False)
        raw_xmap_grid_array[:, :, :] = dataset_dmap_array[:, :, :]
        model_grid = reference_frame.unmask(SparseDMap(model_map))

        # Get the initial events from clustering the Z map
        events, cluster_metadata = ClusterDensityDBSCAN()(z, reference_frame)
        cutoff, high_z_all_points_mask, eps = cluster_metadata.values()
        num_initial_events = len(events)
        if self.debug:
            print(f'model {model_number}: Z map cutoff: {round(cutoff, 2)} results in {num_initial_events} events from {np.sum(high_z_all_points_mask)} high z points and eps {eps}')

        # Handle the edge case of zero events
        if len(events) == 0:
            return None, mean, z, std, {}

        # Filter the events prior to scoring them based on their size
        for filter in [
            FilterSize(reference_frame, min_size=self.minimum_z_cluster_size),
        ]:
            events = filter(events)
        num_size_filtered_events = len(events)


        if self.debug & (num_size_filtered_events > 0):
            size_range = (min([_event.pos_array.shape[0] for _event in events.values()]),
                          max([_event.pos_array.shape[0] for _event in events.values()]))

            print(f'model {model_number}: size filtering results in {num_size_filtered_events} with volume element {round(reference_frame.get_grid().unit_cell.volume / reference_frame.get_grid().point_count,2)} and size range {size_range}')

        # Return None if there are no events after pre-scoring filters
        if len(events) == 0:
            return None, mean, z, std, {}

        # Score the events with some method such as the CNN
        time_begin_score_events = time.time()
        # events = score(ligand_files, events, xmap_grid, raw_xmap_grid, mean_grid, z_grid, model_grid,
        #                median, reference_frame, homogenized_dataset_dmap_array, mean
        #                )
        if self.use_ligand_data:
            for lid, ligand_data in ligand_files.items():
                confs = get_conformers(ligand_data)
                for event_id, event in events.items():
                    conf = set_structure_mean(confs[0], event.centroid)
                    event_score, map_array, mol_array = score(
                        event,
                        conf,
                        z_grid,
                        raw_xmap_grid
                    )
                    if event_score > event.score:
                        event.score = event_score
                    _x, _y, _z, = event.centroid
                    # print(f'\t {model_number}_{event_id}_{lid}: ({_x}, {_y}, {_z}): {round(event_score, 5)}')

                    # dmaps = {
                    #     'zmap': map_array[0][0],
                    #     'xmap': map_array[0][1],
                    #     'mask': mol_array[0][0],
                    # }
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
                    #     ccp4.write_ccp4_map(
                    #         str(fs.output.processed_datasets[dtag] / f'{model_number}_{event_id}_{lid}_{name}.ccp4'))

                time_finish_score_events = time.time()
        else:
            for event_id, event in events.items():
                event_score, map_array, mol_array = score(
                    event,
                    None,
                    z_grid,
                    raw_xmap_grid
                )
                event.score = event_score
                _x, _y, _z, = event.centroid
                event.bdc = get_bdc(event, xmap_grid, mean_grid, protein_density_median)

        # Filter the events after scoring based on keeping only the locally highest scoring event
        num_events = len(events)
        score_range = (round(min([_event.score for _event in events.values()]), 2), round(max([_event.score for _event in events.values()]),2))
        for filter in [
            FilterScore(self.minimum_event_score),  # Filter events based on their score
            # FilterLocallyHighestLargest(self.local_highest_score_radius),  # Filter events that are close to other,
            #                                                                # better scoring events
            # FilterLocallyHighestScoring(self.local_highest_score_radius)
        ]:
            events = filter(events)
        # TODO: Replace with logger printing
        num_score_filtered_events = len(events)

        if self.debug:
            print(f'model {model_number}: score filtering results in {num_score_filtered_events} with cutoff {self.minimum_event_score} and score range {score_range}')


        # Return None if there are no events after post-scoring filters
        if len(events) == 0:
            return None, mean, z, std, {}

        # Renumber the events
        events = {j + 1: event for j, event in enumerate(events.values())}

        # print(f'z map stats: {np.min(z)} {np.max(z)} {np.median(z)} {np.sum(np.isnan(z))}')

        meta = {
            'Number of Initial Events': num_initial_events,
            'Number of Size Filtered Events': num_size_filtered_events,
            'Number of Score Filtered Events': num_score_filtered_events
        }

        return events, mean, z, std, meta


# def score_builds(
#             score_build,
#             builds,
#             raw_xmap_grid,
#             dataset_dmap_array,
#             model_means,
#             model_zs
#         ):
#     for model_number, mean_map in model_means.items():
#         model_z = model_zs[model_number]
#         model_builds = {_build_id: build for _build_id, build in builds.items() if _build_id[0] == model_number}
#
#         # Get the model corrected event maps
#
#         #


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
    # TODO: uses ray not mulyiprocessing_spawn
    console.start_initialise_multiprocessor()
    processor: ProcessorInterface = ProcessLocalRay(args.local_cpus)
    console.print_initialized_local_processor(args)

    # Get the model of the input and output of the program on the file systems
    console.start_fs_model()
    fs: PanDDAFSInterface = PanDDAFS(Path(args.data_dirs), Path(args.out_dir), args.pdb_regex, args.mtz_regex)
    console.summarise_fs_model(fs)

    # Get the method for scoring events
    if args.use_ligand_data:
        event_model_path = Path(os.path.dirname(inspect.getfile(LitEventScoring))) / "model_event.ckpt"
        event_config_path = Path(os.path.dirname(inspect.getfile(LitEventScoring))) / "model_event_config.yaml"
        event_score_quantiles_path = Path(
            os.path.dirname(inspect.getfile(LitEventScoring))) / "event_score_quantiles.csv"
        if not (event_model_path.exists() & event_config_path.exists()):
            print(f'No event model at {event_model_path}. Downloading event model...')
            with open(event_model_path, 'wb') as f:
                # gdown.download('https://drive.google.com/file/d/1b58MUIJdIYyYHr-UhASVCvIWtIgrLYtV/view?usp=sharing',
                #                f)
                gdown.download(id='1b58MUIJdIYyYHr-UhASVCvIWtIgrLYtV',
                               output=f)
            with open(event_config_path, 'wb') as f:
                gdown.download(id='1qyPqPylOguzXmt6XSFaXCKrnvb8gZ8E2',
                               output=f)
            with open(event_score_quantiles_path, 'wb') as f:
                gdown.download(id='15RnkrGtEmFvtBvIlwfaUE1QfQrD2npnu', output=f)

        with open(event_config_path, 'r') as f:
            event_model_config = yaml.safe_load(f)
        score_event_model = load_model_from_checkpoint(
            event_model_path,
            LitEventScoring(event_model_config),
        ).float().eval()
        score_event = EventScorer(score_event_model, event_model_config, debug=args.debug)

        # Get the method for scoring
        build_model_path = Path(os.path.dirname(inspect.getfile(LitBuildScoring))) / "model_build.ckpt"
        build_config_path = Path(os.path.dirname(inspect.getfile(LitBuildScoring))) / "model_build_config.yaml"

        if not (build_model_path.exists() & build_config_path.exists()):
            print(f'No build model at {build_model_path}.Downloading build model...')
            with open(build_model_path, 'wb') as f:
                # gdown.download('https://drive.google.com/file/d/17ow_rxuEvi0LitMP_jTWGMSDt-FfJCkR/view?usp=sharing',
                #                f
                #                )
                gdown.download(id='17ow_rxuEvi0LitMP_jTWGMSDt-FfJCkR',
                               output=f)
            with open(build_config_path, 'wb') as f:
                gdown.download(id='1HEXHZ6kfh92lQoWBHalGbUJ-iCsOIkFo',
                               output=f)
        with open(build_config_path, 'r') as f:
            build_model_config = yaml.safe_load(f)
        score_build_model = load_model_from_checkpoint(
            build_model_path,
            LitBuildScoring(build_model_config),
        ).float().eval()
        score_build = BuildScorer(score_build_model, build_model_config)
        score_build_ref = processor.put(score_build)
        event_score_quantiles = pd.read_csv(event_score_quantiles_path)
    else:  # use_ligand_data=False
        event_model_path = Path(os.path.dirname(inspect.getfile(LitEventScoring))) / "model_event_no_ligand.ckpt"
        event_config_path = Path(
            os.path.dirname(inspect.getfile(LitEventScoring))) / "model_event_no_ligand_config.yaml"
        event_score_quantiles_path = Path(
            os.path.dirname(inspect.getfile(LitEventScoring))) / "event_score_no_ligand_quantiles.csv"
        if not (event_model_path.exists() & event_config_path.exists()):
            print(f'No event model at {event_model_path}. Downloading event model...')
            with open(event_model_path, 'wb') as f:
                gdown.download(id='1ccUM3g6RKluxwz8hofqmXEH2iymMvjyy',
                               output=f)
            with open(event_config_path, 'wb') as f:
                gdown.download(id='1c_QyEjFD5DtYlbU-Gh1o79gkbtdSrkDl',
                               output=f)
            with open(event_score_quantiles_path, 'wb') as f:
                gdown.download(id='1kHtBtLgGBuSBO8Mrf9pn7kjokL6fRMP6', output=f)

        with open(event_config_path, 'r') as f:
            event_model_config = yaml.safe_load(f)
        score_event_model = load_model_from_checkpoint(
            event_model_path,
            LitEventScoring(event_model_config),
        ).float().eval()
        score_event = EventScorer(score_event_model, event_model_config, debug=args.debug)
        event_score_quantiles = pd.read_csv(event_score_quantiles_path)
    if args.debug:
        print(f'Using ligand?: {score_event.model.ligand} / {score_event.model.ligand is True}')
        print(f'Score model path: {event_model_path}')


    # Get the method for processing the statistical models
    process_model = ProcessModel(minimum_event_score=event_model_config['minimum_event_score'], debug=args.debug)

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
    dataset_filters = [
        FilterRFree(args.max_rfree),
        FilterResolutionLowerLimit(args.high_res_lower_limit),
        FilterRange(args.dataset_range),
        FilterExcludeFromAnalysis(args.exclude_from_z_map_analysis),
        FilterOnlyDatasets(args.only_datasets)
    ]
    if args.use_ligand_data:
        dataset_filters.append(FilterNoLigandData())

    datasets_to_process, datasets_not_to_process = GetDatasetsToProcess(dataset_filters)(datasets, fs)
    console.summarize_datasets_to_process(datasets_to_process, datasets_not_to_process)

    # Process each dataset by identifying potential comparator datasets, constructing proposed statistical models,
    # calculating alignments of comparator datasets, locally aligning electron density, filtering statistical models
    # to the plausible set, evaluating those models for events, selecting a model to take forward based on those events
    # and outputing event maps, z maps and mean maps for that model
    pandda_events = {}
    time_begin_process_datasets = time.time()
    console.start_process_shells()
    autobuilds = {}
    for j, dtag in enumerate(datasets_to_process):

        # Record the time that dataset processing begins
        time_begin_process_dataset = time.time()

        # Handle the case in which the dataset has already been processed
        # TODO: log properly
        events_yaml_path = fs.output.processed_datasets[dtag] / f"events.yaml"
        print(f"Checking for a event yaml at: {events_yaml_path}")
        if events_yaml_path.exists():
            print(f"Already have events for dataset! Skipping!")
            _events = serialize.unserialize_events(fs.output.processed_datasets[dtag] / f"events.yaml")
            for event_idx, event in _events.items():
                pandda_events[(dtag, event_idx)] = event
            for event_idx, event in _events.items():
                autobuilds[(dtag, event_idx)] = {
                    event.build.ligand_key: AutobuildResult(
                        {event.build.build_path: {'score': event.build.signal, 'centroid': event.build.centroid}},
                        None, None, None, None, None
                    )
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
            datasets_to_process,
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
                    reference_frame_ref
                )
                for _dtag
                in comparator_datasets
            }
        )
        dmaps = np.vstack([_dmap.data.reshape((1, -1)) for _dtag, _dmap in dmaps_dict.items()])
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
            comparator_datasets,
            dmaps,
            reference_frame,
            CharacterizationNNAndFirst()
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
        plot_aligned_density_projection(
            dmaps,
            models_to_process,
            characterization_set_masks,
            umap_plot_out_dir
        )

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

    time_finish_process_datasets = time.time()
    # TODO: Log properly
    # print(
    #     f"Processed {len(datasets)} datasets in {round(time_finish_process_datasets - time_begin_process_datasets, 2)}")

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
        if args.use_ligand_data & args.autobuild:
            merged_build_scores = merge_autobuilds(
                datasets,
                pandda_events,
                autobuilds,
                fs,
                # MergeHighestRSCC(),
                MergeHighestBuildScore()
                # MergeHighestBuildAndEventScore()
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
    # structure_arrays = {_dtag: StructureArray.from_structure(datasets[_dtag].structure) for _dtag in
    #                         datasets}
    # structure_array_refs = {_dtag: processor.put(StructureArray.from_structure(datasets[_dtag].structure)) for _dtag in
    #                         datasets}
    sites: Dict[int, Site] = get_sites(
        datasets,
        pandda_events,
        datasets_to_process[
            min(
                datasets_to_process,
                key=lambda _dtag: datasets_to_process[_dtag].reflections.resolution()
            )
        ],
        # processor,
        # structure_arrays,
        # structure_array_refs,
        HeirarchicalSiteModel(t=args.max_site_distance_cutoff),
        # ResidueSiteModel()
    )
    # TODO: Log properly
    # print("Sites")
    # for site_id, site in sites.items():
    #     print(f"{site_id} : {site.centroid} : {site.event_ids}")

    # Rank the events for display in PanDDA inspect
    ranking = rank_events(
        pandda_events,
        sites,
        autobuilds,
        RankHighEventScoreBySite(),
        # RankHighEventScore(),
        # RankHighBuildScore()
        # RankHighEventBuildScore()
    )
    # for event_id in ranking:
    #     print(f"{event_id} : {round(pandda_events[event_id].build.score, 2)}")

    # Probabilities
    # Calculate the cumulative probability that a hit remains in the site using the event score quantile table
    hit_in_site_probabilities = get_hit_in_site_probabilities(pandda_events, ranking, sites, event_score_quantiles)

    # Output the event and site tables
    output_tables(datasets, pandda_events, ranking, sites, hit_in_site_probabilities, fs)
    time_pandda_finish = time.time()
    # TODO: Log properly
    print(f"PanDDA ran in: {round(time_pandda_finish - time_pandda_begin, 2)} seconds!")
