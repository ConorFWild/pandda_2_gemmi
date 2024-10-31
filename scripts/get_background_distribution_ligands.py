import itertools
import os
import shutil
import time
import inspect

from sklearnex import patch_sklearn

patch_sklearn()

import numpy as np
import gemmi
from statsmodels.stats.diagnostic import lilliefors
import diptest
import yaml
import pandas as pd

from pandda_gemmi.interfaces import *
from pandda_gemmi import constants
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
    FilterRange,
    FilterExcludeFromAnalysis,
    FilterOnlyDatasets,
    FilterSpaceGroup,
    FilterResolution,
    FilterCompatibleStructures,
    FilterResolutionLowerLimit
)
from pandda_gemmi.event_model.characterization import get_characterization_sets, CharacterizationFirst
from pandda_gemmi.event_model.filter_characterization_sets import filter_characterization_sets
from pandda_gemmi.event_model.outlier import PointwiseNormal, PointwiseMAD
from pandda_gemmi.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.event_model.score import get_model_map
from pandda_gemmi.event_model.filter import (
    FilterSize,
    FilterScore,
)

from pandda_gemmi.autobuild import AutobuildResult
from pandda_gemmi.autobuild.inbuilt import get_conformers
from pandda_gemmi.pandda_logging import PanDDAConsole
from pandda_gemmi import serialize
from pandda_gemmi.cnn import load_model_from_checkpoint, EventScorer, LitEventScoring, BuildScorer, LitBuildScoring, \
    set_structure_mean


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
                 minimum_event_score=0.01,
                 local_highest_score_radius=8.0
                 ):
        self.minimum_z_cluster_size = minimum_z_cluster_size
        self.minimum_event_score = minimum_event_score
        self.local_highest_score_radius = local_highest_score_radius

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
        print(f'z map stats: {np.min(z)} {np.max(z)} {np.median(z)} {np.sum(np.isnan(z))}')

        mean_grid = reference_frame.unmask(SparseDMap(mean))
        z_grid = reference_frame.unmask(SparseDMap((z - np.mean(z)) / np.std(z)))

        # Get the median
        median = np.median(mean[reference_frame.mask.indicies_sparse_inner_atomic])

        # unsparsify input maps
        xmap_grid = reference_frame.unmask(SparseDMap(homogenized_dataset_dmap_array))
        raw_xmap_grid = gemmi.FloatGrid(*dataset_dmap_array.shape)
        raw_xmap_grid.set_unit_cell(z_grid.unit_cell)
        raw_xmap_grid_array = np.array(raw_xmap_grid, copy=False)
        raw_xmap_grid_array[:, :, :] = dataset_dmap_array[:, :, :]
        model_grid = reference_frame.unmask(SparseDMap(model_map))

        # Get the initial events from clustering the Z map
        events, cutoff = ClusterDensityDBSCAN()(z, reference_frame)

        # Handle the edge case of zero events
        if len(events) == 0:
            return None, mean, z, std

        # Filter the events prior to scoring them based on their size
        for filter in [
            FilterSize(reference_frame, min_size=self.minimum_z_cluster_size),
        ]:
            events = filter(events)

        # Return None if there are no events after pre-scoring filters
        if len(events) == 0:
            return None, mean, z, std

        # Score the events with some method such as the CNN
        time_begin_score_events = time.time()
        # events = score(ligand_files, events, xmap_grid, raw_xmap_grid, mean_grid, z_grid, model_grid,
        #                median, reference_frame, homogenized_dataset_dmap_array, mean
        #                )
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
                event.score = event_score
                _x, _y, _z, = event.centroid
                print(f'\t {model_number}_{event_id}_{lid}: ({_x}, {_y}, {_z}): {round(event_score, 5)}')

                dmaps = {
                    'zmap': map_array[0][0],
                    'xmap': map_array[0][1],
                    'mask': mol_array[0][0],
                }
                for name, dmap in dmaps.items():
                    grid = gemmi.FloatGrid(32, 32, 32)
                    uc = gemmi.UnitCell(16.0, 16.0, 16.0, 90.0, 90.0, 90.0)

                    # uc = gemmi.UnitCell(8.0, 8.0, 8.0, 90.0, 90.0, 90.0)
                    grid.set_unit_cell(uc)

                    grid_array = np.array(grid, copy=False)
                    grid_array[:, :, :] = dmap[:, :, :]
                    ccp4 = gemmi.Ccp4Map()
                    ccp4.grid = grid
                    ccp4.update_ccp4_header()
                    ccp4.write_ccp4_map(
                        str(fs.output.processed_datasets[dtag] / f'{model_number}_{event_id}_{lid}_{name}.ccp4'))

            time_finish_score_events = time.time()

        # Filter the events after scoring based on keeping only the locally highest scoring event
        num_events = len(events)
        for filter in [
            FilterScore(self.minimum_event_score),  # Filter events based on their score
            # FilterLocallyHighestLargest(self.local_highest_score_radius),  # Filter events that are close to other,
            #                                                                # better scoring events
            # FilterLocallyHighestScoring(self.local_highest_score_radius)
        ]:
            events = filter(events)
        # TODO: Replace with logger printing
        print(f"CNN Filtered {num_events} down to {len(events)}")

        # Return None if there are no events after post-scoring filters
        if len(events) == 0:
            return None, mean, z, std

        # Renumber the events
        events = {j + 1: event for j, event in enumerate(events.values())}

        print(f'z map stats: {np.min(z)} {np.max(z)} {np.median(z)} {np.sum(np.isnan(z))}')

        return events, mean, z, std


def get_lilliefors_map(characterization_set_dmaps_array):
    n_datasets, n_datapoints = characterization_set_dmaps_array.shape

    lilliefors_map = np.zeros(n_datapoints)

    for n in range(n_datapoints):
        ps = characterization_set_dmaps_array[:, n].flatten()
        if np.any(ps > 0):
            stat, pval = lilliefors(ps)
            lilliefors_map[n] = pval

    return lilliefors_map


def get_dip_map(characterization_set_dmaps_array):
    n_datasets, n_datapoints = characterization_set_dmaps_array.shape

    dip_map = np.zeros(n_datapoints)

    for n in range(n_datapoints):
        ps = characterization_set_dmaps_array[:, n].flatten()
        if np.any(ps > 0):
            dval, pval = diptest.diptest(ps)
            dip_map[n] = pval

    return dip_map


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

    # Get the method for scoring events
    score_event_model = load_model_from_checkpoint(
        Path(os.path.dirname(inspect.getfile(LitEventScoring))) / "model_event.ckpt",
        LitEventScoring(),
    ).eval()
    score_event = EventScorer(score_event_model)

    # Get the method for scoring
    score_build_model = load_model_from_checkpoint(
        Path(os.path.dirname(inspect.getfile(LitBuildScoring))) / "model_build.ckpt",
        LitBuildScoring(),
    ).eval()
    score_build = BuildScorer(score_build_model)
    score_build_ref = processor.put(score_build)

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
    datasets_to_process, datasets_not_to_process = GetDatasetsToProcess(
        [
            FilterRFree(args.max_rfree),
            FilterResolutionLowerLimit(args.high_res_lower_limit),
            FilterRange(args.dataset_range),
            FilterExcludeFromAnalysis(args.exclude_from_z_map_analysis),
            FilterOnlyDatasets(args.only_datasets)
        ]
    )(datasets, fs)
    console.summarize_datasets_to_process(datasets_to_process, datasets_not_to_process)

    # Process each dataset by identifying potential comparator datasets, constructing proposed statistical models,
    # calculating alignments of comparator datasets, locally aligning electron density, filtering statistical models
    # to the plausible set, evaluating those models for events, selecting a model to take forward based on those events
    # and outputing event maps, z maps and mean maps for that model
    pandda_events = {}
    time_begin_process_datasets = time.time()
    console.start_process_shells()
    autobuilds = {}

    # Get the datasets with modelled ligands in the source PanDDA
    print('Getting ligands!!')
    inspect_table = pd.read_csv(args.source_pandda / 'analyses' / 'pandda_inspect_events.csv')
    print(f'Read csv...')
    ligand_models = {}
    for _idx, _row in inspect_table.iterrows():
        print(_row['Ligand Confidence'])
        if _row['Ligand Confidence'] != 'High':
            continue
        dtag = _row['dtag']
        dataset_dir = args.source_pandda / 'processed_datasets' / dtag
        modelled_structure_path = dataset_dir / 'modelled_structures' / constants.PANDDA_EVENT_MODEL.format(dtag)

        modelled_structure = gemmi.read_structure(str(modelled_structure_path))
        print(modelled_structure_path)

        for model in modelled_structure:
            for chain in model:
                for res in chain:
                    if res.name in ['LIG', 'XXX']:
                        ligand_models[dtag] = res

    print('Got ligands')
    print(ligand_models)

    print('Processing Datasets!')
    for j, dtag in enumerate(datasets_to_process):
        print(f'Processing {dtag}')
        if dtag not in ligand_models:
            print(f'\tSkipping {dtag}')
            continue

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
            return

        # Get the dataset
        dataset = datasets[dtag]

        # Skip processing the dataset if there is no ligand data
        if len([_key for _key in dataset.ligand_files if dataset.ligand_files[_key].ligand_cif]) == 0:
            console.no_ligand_data()
            return

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
            return

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
        print(f"\t\tGot alignments in: {round(time_finish_get_alignments - time_begin_get_alignments, 2)}")

        # Get the reference frame and save it to the object store
        time_begin_get_frame = time.time()
        reference_frame: DFrame = DFrame(dataset, processor)
        reference_frame_ref = processor.put(reference_frame)
        time_finish_get_frame = time.time()
        # TODO: Log properly
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
        # TODO: log properly
        print(f"\t\tGot dmaps in: {round(time_finish_get_dmaps - time_begin_get_dmaps, 2)}")
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
            CharacterizationFirst()
        )
        time_finish_get_characterization_sets = time.time()
        # TODO: Log properly
        print(
            f"\t\tGot characterization sets in: {round(time_finish_get_characterization_sets - time_begin_get_characterization_sets, 2)}")

        models_to_process, model_scores, characterization_set_masks = filter_characterization_sets(
            comparator_datasets,
            characterization_sets,
            dmaps,
            dataset_dmap_array,
            reference_frame,
            PointwiseMAD(),
            process_all=True
        )

        # Get the model maps
        print(characterization_set_masks)
        characterization_set_dmaps_array = dmaps[characterization_set_masks[1], :]

        # Get the mean and STD
        mean, std, z = PointwiseNormal()(
            dataset_dmap_array,
            characterization_set_dmaps_array
        )
        print(f'z map stats: {np.min(z)} {np.max(z)} {np.median(z)} {np.sum(np.isnan(z))}')

        mean_grid = reference_frame.unmask(SparseDMap(mean))
        z_grid = reference_frame.unmask(SparseDMap((z - np.mean(z)) / np.std(z)))

        # Get the lilliefors map
        print('Calculating lilliefors statistics...')
        lilliefors_map = get_lilliefors_map(characterization_set_dmaps_array)

        # Dip map
        print('Calculating dip statistics...')
        dip_map = get_dip_map(characterization_set_dmaps_array)

        # Sample atom positions in ground state maps
        print('Calculating samples...')
        samples = {}
        for characterization_dtag, ground_state_dmap_array in zip(characterization_sets[1], characterization_set_dmaps_array, ):
            ground_state_dmap = reference_frame.unmask(SparseDMap(ground_state_dmap_array))

            samples[characterization_dtag] = {}
            for atom in ligand_models[dtag]:
                pos = atom.pos
                val = ground_state_dmap.interpolate_value(pos)
                samples[characterization_dtag][atom.name] = val

        # Delete other content and save
        shutil.rmtree(args.out_dir)
        os.mkdir(args.out_dir)
        with open(Path(args.out_dir) / f'{dtag}_lilliefors.npy', 'wb') as f:
            np.save(f, lilliefors_map, )
        with open(Path(args.out_dir) / f'{dtag}_dip.npy', 'wb') as f:
            np.save(f, dip_map, )
        with open(Path(args.out_dir) / f'{dtag}_samples.yaml', 'w') as f:
            yaml.dump(samples, f)

    print(f'Finished!')

if __name__ == '__main__':
    # Parse Command Line Arguments
    args = PanDDAArgs.from_command_line()

    pandda(args)
